# Kalman tracker and control loop for the Pinpoint turret

> [!NOTE]
> **Status (2026-11-25):** the research behind the tracker plan, which is built in **Phase 2** (see [ROADMAP.md](../ROADMAP.md)). Where the team decided differently, the ROADMAP wins:
> - **In scope for us:** §1–3 (camera geometry, Kalman filter, association) and target selection (§4.6, simplified: sticky target, then the nearest confirmed track).
> - **Not ours for now:** §4.1–4.3 and build steps 6, 7 and 10 (PID, plant simulator, serial and hardware integration). The gimbal team owns the motor loop unless they tell us otherwise.
> - **Waiting on launcher numbers:** §4.4–4.5 and step 9 (lead, dart drop, parallax, fire gate, pop confirmation).
> - **Coasting:** confirmed tracks coast for **1 s** before deletion (not 10 frames).
> - **Hardware:** the host is most likely a Raspberry Pi, not a laptop. The C270 is **fixed focus** (Logitech spec), so the "lock autofocus" advice in §1 and §8 doesn't apply.

Researched 2026-11-25. Context: C270 on the gimbal (camera moves with the barrel), 1280x720 at 30 fps, balloons 3 to 6 m away, several per frame, boxes from classical OpenCV or YOLO, sub-degree aim, host laptop to MCU over USB serial. Team rule: Kalman filter and control loop are hand-written (NumPy only).

## 0. Source verification status (read first)

Fetched and read this session:
- SORT paper, Bewley et al. 2016, via [ar5iv](https://ar5iv.labs.arxiv.org/html/1602.00763) (arXiv 1602.00763): Sections 3.2 to 3.4.
- Labbe, *Kalman and Bayesian Filters in Python*, ch. 7 (F, Q, predict/update, Joseph form): [raw notebook](https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/07-Kalman-Filter-Math.ipynb). Ch. 8 only partly read.
- Logitech C270 product page: [logitech.com](https://www.logitech.com/en-us/shop/p/c270-hd-webcam.960-000694).
- Caltech Feedback Systems PID page: [PID_Control.html](https://www.cds.caltech.edu/~murray/amwiki/PID_Control.html). It confirms windup and the anti-windup feedback idea but its equations are images, so the PID equations below are standard textbook forms, not read from the source.

Could NOT be fetched (403/404/PDF unreadable). Claims attributed to them below come from background knowledge and need a check against the source:
- OpenCV calibration tutorial / `calib3d` docs (403).
- Chaumette and Hutchinson, "Visual servo control, Part I: Basic approaches", IEEE RAM 2006 (blocked).
- Astrom and Murray, *Feedback Systems* 2nd ed., PID chapter PDF (binary, unreadable by the fetcher). Chapter numbering differs between editions.

Items marked ASSUMPTION are engineering estimates to be measured on hardware.

## 1. Camera geometry (pixel to angle)

**C270 field of view is 55 degrees diagonal, not about 60.** Logitech: "Crisp HD 720p/30 fps video calls with diagonal 55 degree field of view" ([product page](https://www.logitech.com/en-us/shop/p/c270-hd-webcam.960-000694)). Other listings quote 60 degrees; trust the manufacturer page, then calibrate.

Pinhole model (OpenCV calib3d convention): `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`, camera matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]] plus distortion (k1,k2,p1,p2,k3).

Prior from the 55 degree spec at 1280x720 (computed here, assuming square pixels, centred principal point, and the full 16:9 frame used):
- diagonal = 1468.6 px, so f = (1468.6/2)/tan(27.5 deg) = **about 1411 px** (fx = fy).
- HFOV about 48.8 deg, VFOV about 28.6 deg.
- About 0.0406 deg/px, or **24.6 px per degree** at the image centre. Pixel quantisation is not the accuracy limit; latency, gimbal backlash and calibration are.
- Target size: a 30 cm balloon at 5 m subtends 3.4 deg (about 83 px). The hit tolerance is the balloon radius: 2.9 deg at 3 m, 1.4 deg at 6 m, plus dart dispersion. So "sub-degree" is the error budget for the 6 m case.
- Barrel/camera parallax: with an ASSUMPTION of 5 cm offset, 0.95 deg at 3 m and 0.48 deg at 6 m. Compensate using range from box size, `Z = f*D/w_px` with D the known balloon diameter. This is the same size of error as the whole budget, so do not skip it.

Calibration: `cv2.calibrateCamera` with 10 to 20+ chessboard views at fixed focus, 1280x720; check the returned RMS reprojection error (below about 0.5 px is good; per the OpenCV docs, background knowledge, unverified). Lock autofocus and auto-exposure off in capture, since a C270 focus change alters f. Use `cv2.undistortPoints` on box centres (cheap, only the centre point, not the image).

Pixel to angle (camera frame), with undistorted normalised coordinates:
```
x = (u - cx)/fx ;  y = (v - cy)/fy          # normalised
ray_c = normalize([x, y, 1])
az_c = atan2(x, 1)        # right positive
el_c = atan2(-y, hypot(x,1))   # up positive
```
Within +/-14 deg the small-angle form `az_c = x`, `el_c = -y` has under 1 percent error, but use the exact form; it is one line.

## 2. Kalman filter design

### 2.1 Which space: world angles, not pixels

The camera moves with the barrel. A balloon that is stationary in the room moves in the image at `-f*omega` whenever the gimbal slews. SORT explicitly "does not account for camera motion" (SORT sec. 3.2), and its constant-velocity model is for a static camera. In pixel space the filter would see the slew as balloon velocity, and the velocity estimate would blow up and then lag when the gimbal stops.

Recommended: track in **world (gimbal-base) angles**:
```
az_w = pan(t_capture)  + az_c        # az_c from the box centre, section 1
el_w = tilt(t_capture) + el_c
```
Exact composition is a rotation R_pan(pan) * R_tilt(tilt) * ray_c, then atan2; the additive form is fine while the tilt is small, or when the combined error under 0.1 deg is acceptable. Use the rotation version if tilt exceeds about 20 deg (ASSUMPTION, check numerically in the simulator).

Consequences:
- Balloons are nearly static in the room (helium balloons drift slowly indoors, ASSUMPTION), so world-angle tracks have near-zero velocity. The turret motion is separated out entirely; the control loop owns the gimbal, the tracker owns the world.
- It needs the **gimbal angle at the exposure time**, not at processing time. Timestamp each frame at grab (`time.monotonic()` right after `cap.read()` returns, minus a measured camera latency), and keep a ring buffer of (t, pan, tilt) from the controller. Interpolate to `t_capture`. Steppers have no feedback, so "angle" is the commanded or dead-reckoned step count; servos have none either. Backlash and missed steps are un-modelled error; fold them into R (below).
- During fast slews the image is motion-blurred and the box shifts. Inflate R with `|omega|` (e.g. `R = R0 + (k*|omega|*exposure)^2`), or drop the update when `|omega|` exceeds a threshold.
- Pan range 120 to 180 degrees means no +/-180 wrap problem. Do not use a wrapping angle library; plain floats are fine.

Pixel-space tracking is acceptable only for pre-hardware replay of a static camera (then world angle equals camera angle plus a constant), which is exactly the recorded-footage case.

### 2.2 Constant velocity vs constant acceleration

Use **constant velocity (CV)**, one 2-state filter per axis (az, el), state `x = [theta, theta_dot]`. Reasons: balloons are slow and near-static; a 3-state constant-acceleration (CA) filter amplifies measurement noise in the acceleration estimate at 30 fps and adds tuning burden. Move to CA only if the innovation test (2.5) shows persistent lag. SORT also uses CV ([sec. 3.2](https://ar5iv.labs.arxiv.org/html/1602.00763)). Labbe ch. 7 gives both models; the CA matrices are below for reference.

Also keep a scalar track of box width (or log width) for range; a simple exponential filter is enough, or add it as a third decoupled 1-state KF.

### 2.3 Matrices (from Labbe ch. 7)

Per axis, dt in seconds, measurement `z = theta_meas` (world angle):
```
F  = [[1, dt],            H = [1, 0]
      [0, 1 ]]

Q  = sigma_a^2 * [[dt^4/4, dt^3/2],      # piecewise white noise, Labbe ch.7
                  [dt^3/2, dt^2  ]]      # sigma_a = std of unmodelled accel (deg/s^2)

R  = sigma_z^2            (1x1 per axis)
```
CA version for reference: `F = [[1,dt,dt^2/2],[0,1,dt],[0,0,1]]`, `Q = sigma^2*[[dt^4/4,dt^3/2,dt^2/2],[dt^3/2,dt^2,dt],[dt^2/2,dt,1]]`, `H=[1,0,0]`.

Predict / update (Labbe ch. 7):
```
x- = F x            P- = F P F^T + Q
S  = H P- H^T + R   K  = P- H^T S^-1
y  = z - H x-       x  = x- + K y
P  = (I-KH) P- (I-KH)^T + K R K^T       # Joseph form, numerically safer
```
Az and el can share one 4-state filter with block-diagonal F, Q, R; separate scalar-measurement filters are simpler to hand-write and need no matrix inverse (S is 1x1). A single 4-state filter with a 2x2 S is needed only if you want a joint Mahalanobis gate (2.4); for hand-written code, build one 4-state filter and invert the 2x2 S explicitly.

### 2.4 Variable dt

Do not hard-code 1/30. Recompute `F(dt)` and `Q(dt)` every step with `dt = t_capture[k] - t_capture[k-1]` from the frame timestamps. Missed frames and detector stalls then enlarge Q automatically, and the same code predicts forward by any horizon `T` (used for lead, section 4). Guard: clamp dt to e.g. [1 ms, 0.5 s]; if a gap exceeds the delete threshold the track is dead anyway.

### 2.5 Tuning Q and R

- `sigma_z` (R): world-angle measurement error = detector box-centre jitter (a few px, 2 px is about 0.08 deg) plus gimbal-angle error (stepper: negligible per step, backlash and lost steps dominate; hobby servo: unknown, no feedback, likely 0.5 deg or worse; ASSUMPTION). Start at 0.15 deg; measure the jitter of box centres on a stationary balloon in recorded footage (std of centre over 300 frames), then add the gimbal term.
- `sigma_a` (Q): the std of balloon acceleration you refuse to model. For drifting balloons start about 5 deg/s^2 at 5 m (ASSUMPTION), sweep over 0.5 to 50 on a log scale.
- Evaluate with the **normalised innovation squared**: `NIS = y^T S^-1 y`. For a consistent filter NIS averaged over time is about the measurement dimension (1 per axis, or 2 jointly). If NIS is above that persistently, Q is too small (lag); below, Q is too large (noisy). Labbe ch. 8 discusses filter evaluation (only partly read this session; verify the NIS/NEES section).
- Tuning loop is offline: replay detections, compute mean NIS and RMS of one-step-ahead prediction error across a grid of (sigma_a, sigma_z).
- Initial covariance: `P0 = diag(sigma_z^2, large)` for velocity, as SORT does ("velocity covariance initialised to large values", sec. 3.4).

## 3. Data association (several balloons)

SORT (Bewley et al. 2016) is the reference: per-frame, predict each track, build cost matrix, solve with the Hungarian algorithm, threshold by IOU_min, spawn tracks for unmatched detections, delete tracks lost for T_Lost frames (sec. 3.3, 3.4; verified). Specifics from the paper: cost is IOU distance between detections and predicted boxes; unmatched detections with low overlap start new tracks with zero velocity and large velocity covariance; new tracks sit in a probationary period; **T_Lost = 1** in their experiments; a lost then reappearing object gets a new identity.

Adaptations for this turret:
- **Cost in angle space, not IOU.** Balloons are small (40 to 65 px, from the repo's YOLO notes), and a 2 px shift changes IOU a lot; box centres are also what you aim at. Use the squared Mahalanobis distance between each detection (world angles) and each track prediction, `d2 = y^T S^-1 y`, as the cost. This also uses the track uncertainty. Fallback: Euclidean angular distance.
- **Gate** with `d2 < 9.21` (chi-square 99 percent, 2 DOF; standard statistics, not from a fetched source). Pairs outside the gate get cost = infinity (large constant).
- **Assignment**: with at most about 6 balloons, a hand-written solution is cheap: either exhaustive search over permutations (6! = 720, trivial) or greedy nearest-neighbour. Greedy NN can mis-assign when two balloons are close (a pair of balloons 10 cm apart at 5 m is 1.1 deg, i.e. 27 px, within a plausible gate), so prefer exact assignment. Hungarian is `scipy.optimize.linear_sum_assignment` if the team accepts it as the one allowed library; write the brute-force version first and use scipy as the test oracle.
- **Birth**: unmatched detection with score above a threshold starts a *tentative* track (zero velocity, large P_vel).
- **Confirmation**: tentative to confirmed after M hits in the last N frames (start 3 of 5, ASSUMPTION; SORT's value is not given in sec. 3.4, the paper just says "probationary").
- **Deletion**: coast (predict only) while missed; delete after `T_lost` consecutive misses. SORT's T_Lost = 1 is for pedestrians with ID continuity; here a balloon is static in the world, so coasting is safe and more misses can be tolerated: start 10 frames (about 0.33 s), tune on recorded dropouts. Tentative tracks delete on the first miss.
- **Do not shoot unconfirmed tracks.** False positives (a red object on a shelf) will otherwise cost darts.
- Ego-motion bonus: since tracks live in world angles, no camera-motion compensation is needed in association.

## 4. Control

### 4.1 Image-based visual servoing (IBVS) with the camera on the barrel

Chaumette and Hutchinson (2006), Part I (not fetched; from background knowledge, verify): in an eye-in-hand setup the feature error is `e = s - s*`, the feature rate relates to camera twist by `s_dot = L_s v_c`, and the proportional law is `v_c = -lambda * L_s^+ e` giving exponentially decaying error `e_dot = -lambda e` for the ideal model. For a single image point and a camera that only rotates (pan/tilt, no translation), the interaction matrix reduces to roughly `L = [[xy, -(1+x^2), y], [1+y^2, -xy, -x]]` on the rotational part, which near the image centre is approximately `-f` times identity. So:

```
omega_pan  =  lambda * (u - u*) / fx
omega_tilt = -lambda * (v - v*) / fy        # sign depends on axis convention
```
where `s* = (u*, v*)` is the image pixel where the barrel actually points (the boresight point, found by calibration, corrected for parallax from section 1). Because the camera and barrel are rigidly mounted, aiming the barrel is just driving the target to `s*`. This is plain proportional control on angular error, which is exactly what the pan/tilt loop below does.

Two ways to feed the loop:
1. **Image error (pure IBVS)**: error computed from the current frame's box, no tracker. Simple, but noisy and delayed.
2. **Track-based (recommended)**: error in world angles from the Kalman track, `e = theta_target_pred - theta_gimbal_now`. Because the track is in world angles, the loop works across frames and during detector dropouts, and the KF can predict the target at the time the command takes effect.

### 4.2 Command type: stepper vs servo

- **Stepper (open loop)**: velocity-command is natural (send step rate, microcontroller generates pulses with a trapezoid/S-curve ramp). The plant is an integrator, so a P term alone gives zero steady-state error to a static target; add the KF velocity as feedforward for moving targets. Position-command (send a target step count) also works and keeps the angle bookkeeping exact; the microcontroller handles acceleration. Choose by who owns the ramp: keep rate and acceleration limits on the MCU (it has the hard timing), and also clamp on the host so the model matches.
- **Hobby servo**: position-command only (angle to PWM). The servo's internal loop supplies the position feedback. Command `theta_cmd = theta_cmd_prev + K*e` (incremental), clamp slew rate on the host. Hobby servos have a PWM deadband and coarse resolution (general knowledge: often around 1 us, a fraction of a degree; ASSUMPTION), and no angle readback, so the "gimbal angle" the tracker sees is the commanded one. For sub-degree work this is the risky choice; flag to the motor team.
- Either way, keep the host-side gimbal model (angle estimate) in a class that the simulator can replace.

### 4.3 P, PI, PID

Standard forms (Astrom and Murray, *Feedback Systems*, PID chapter; the equations above were not extractable from the fetched page, so confirm against the PDF):
```
u = Kp*e + Ki * integral(e) + Kd * d(e)/dt
```
Recommendations for this plant:
- **Start with P (velocity command)**: plant is an integrator (stepper velocity) or already closed-loop (servo). P gives no steady-state error to constant targets.
- **Add feedforward (KF velocity)** before adding I: `omega_cmd = Kp*e + theta_dot_target`. This is what removes tracking lag on drifting balloons.
- **Add I only if** a measured steady offset appears (gimbal bias, parallax mismatch, servo deadband). Clamp the integrator, and only integrate when `|e|` is below a window (e.g. 2 deg), since the integrator should fix residual offset, not slew error.
- **Derivative on measurement, not on error**: avoid a derivative kick when the setpoint (selected target) jumps to a new balloon. Here the "measurement" is the gimbal angle, which is smooth. Even better: the KF already gives you target velocity; use the gimbal rate for damping instead of differentiating noisy errors. If D is used, low-pass it (first-order filter, N about 5 to 20; standard setpoint-weighting and filtered-derivative forms per Astrom and Murray, verify).
- **Anti-windup**: back-calculation, `I_dot = Ki*e + (u_sat - u)/Tt`, with `Tt` about `sqrt(Ti*Td)` or simply Ti/5 (Caltech's wiki confirms the idea: feed back the difference between commanded and actual actuator input; the exact formula is from standard texts). The simplest robust alternative is conditional integration: freeze the integrator whenever the output saturates or the rate limit is active.
- **Deadband**: stop commanding when `|e| < e_db` (e.g. 0.1 to 0.2 deg) to avoid dithering a stepper or chattering a servo's PWM; keep it smaller than the fire-gate threshold.
- **Rate and acceleration limits**: clamp `|omega| <= omega_max` and `|d omega/dt| <= alpha_max` (set from motor torque and the camera blur budget). Apply the limiter to the command before it goes to the serial link, and let the simulator use the same limits.
- Suggested starting gain: `Kp` so that the closed-loop time constant is about 0.15 to 0.25 s (`omega = e/tau`, i.e. Kp = 1/tau about 4 to 7 1/s), subject to the delay below. Delay limits it: a pure delay `L` gives a stability limit near `Kp*L < pi/2`; with `L` of 0.15 s (ASSUMPTION), Kp must stay under about 10 1/s, and a good margin is half that.

### 4.4 Latency, prediction, lead

Latency chain (all ASSUMPTION until measured; C270 has no hardware timestamp):
camera exposure+USB+driver (about 60 to 120 ms) + inference (5 to 30 ms) + tracker (<1 ms) + serial and MCU (about 2 to 10 ms) + actuator response. Measure end-to-end with a blinking LED or a screen timestamp test, and record `L_cam`, the capture-to-host delay, as a parameter used to back-date the timestamps in 2.1.

Use the filter to compensate:
```
t_aim = t_now + L_actuate + T_flight(Z)         # when the dart will actually arrive
theta_aim = predict(track, t_aim)               # KF predict(), same F(dt) code
```
- The estimate refers to the instant of exposure `t_capture`; predict from there to `t_now` (catch up the latency) then to `t_aim`.
- **Dart flight time**: `T = Z / v_dart`. ASSUMPTION v about 15 to 25 m/s for a foam dart (chronograph it): about 0.2 to 0.4 s at 3 to 6 m. Over 0.3 s a balloon drifting 5 cm/s moves 1.5 cm, negligible; the lead mainly matters for gimbal motion itself and fast-moving balloons.
- **Ballistic drop is larger than the lead**: `0.5*g*T^2` = 0.31 m at T = 0.25 s, i.e. about 3.5 deg at 5 m. Boresight at one range and add a range-dependent elevation offset from a lookup table measured on a test range (`el_offset(Z)`), with Z from box size.
- Keep prediction horizons short: covariance grows with `T`; if the predicted `P_theta` at `t_aim` exceeds a limit, hold fire.

### 4.5 Fire gate

Fire only when all hold:
1. Selected track is confirmed and its `sqrt(P_theta)` (predicted at `t_aim`) is below about 0.3 deg.
2. Angular error `|theta_aim - theta_gimbal|` < `e_fire` (start at 0.5 * balloon angular radius, bounded by 0.5 deg) for **N consecutive frames** (start N = 5, about 150 ms), using the gimbal angle at command time. The error counter resets on any frame above threshold or any lost measurement.
3. Gimbal rate below a limit (not mid-slew), range inside 3 to 6 m, dart-ready and safety interlocks on the MCU side.
4. Minimum time since the last shot (reload, spin-up).

### 4.6 Target selection and pop confirmation

Selection:
- Only confirmed tracks in range.
- Cost = angular slew distance from the current aim plus a penalty for large uncertainty and a bias for tracks already engaged (hysteresis) so the turret does not flip between two near-equal balloons. Switch only if another track is better by a margin (e.g. 1.5x), or the current one is lost or resolved.
- Hold the selection (sticky) until hit, miss-timeout, or loss.

Pop confirmation (a track vanishes right after a fire):
- On fire at `t_f`, record the engaged track and its predicted state; open a window `[t_f + T_flight - dt_margin, t_f + T_flight + 0.5 s]`.
- **Hit** = engaged track stops being matched (consecutive misses reach `T_lost`) inside the window, and no detection reappears within the gate for about 0.3 s. Optionally require a supporting cue: sudden size or colour-fragment change on the last frames, or other tracks unaffected.
- **Miss** = track still matched after the window closes. Re-engage, with a retry limit, and add a lateral correction if misses are systematic.
- Disappearance outside the window, or while the dart was never fired, is a lost track (occlusion, detector dropout) and is not scored as a hit.
- Mark the dead track as "popped" for a cool-down (a few seconds) so a late ghost detection at the same angle does not spawn an immediate re-target.
- Scoring hits only from the track vanishing is the cheapest design; if the detector sometimes drops balloons for a few frames, a longer coast time and the window above prevent false hits.

## 5. Pseudocode

```python
# per-frame, host side. All angles in degrees, time in seconds.
def on_frame(frame, t_grab):
    t_cap = t_grab - L_CAM                       # back-date to exposure
    pan, tilt = gimbal_log.interp(t_cap)         # angle at exposure time
    dets = detector(frame)                       # [(x1,y1,x2,y2,score), ...]
    meas = [to_world(box, pan, tilt, K, dist) for box in dets]  # (az, el, width_px)

    for tr in tracks: tr.predict(t_cap)          # F(dt), Q(dt), dt from timestamps
    pairs, un_d, un_t = associate(tracks, meas)  # Mahalanobis cost, gate 9.21, exact assign
    for tr, m in pairs: tr.update(m, R_of(omega_now)); tr.hits += 1; tr.miss = 0
    for tr in un_t:     tr.miss += 1
    for m in un_d:      tracks.append(Track(m, state="tentative"))
    tracks = [t for t in tracks if not t.dead()]  # tentative: 1 miss; confirmed: T_LOST misses

    tgt = select(tracks, gimbal_now)              # confirmed, in range, hysteresis
    if tgt:
        z = range_from_width(tgt)
        t_aim = t_cap_to_now_plus(L_ACT + z / V_DART)
        th = tgt.predict_copy(t_aim)              # (az, el) + sigma
        th.el += el_offset(z) + parallax(z)       # ballistic drop and barrel offset
        ctrl_step(th)
        fire_gate(tgt, th)

def ctrl_step(th):                                # runs on each new aim estimate
    e = th - gimbal_cmd_angle
    e = 0 if abs(e) < DEADBAND else e
    w = KP * e + th.velocity                      # P + KF velocity feedforward
    w = clamp(w, -W_MAX, W_MAX)
    w = slew_limit(w, w_prev, A_MAX * dt)         # acceleration limit
    send_serial(w)                                # or integrate to a position command for servos

def fire_gate(tgt, th):
    ok = (tgt.confirmed and tgt.sigma(th) < SIG_MAX and abs(err) < E_FIRE
          and abs(gimbal_rate) < RATE_MAX and Z_MIN <= z <= Z_MAX)
    ok_count = ok_count + 1 if ok else 0
    if ok_count >= N_FIRE and cooldown_over():
        send_fire(); pending = (tgt.id, t_now, window(z))
```

## 6. Offline testing

1. **Replay harness.** Record per frame: capture timestamp, frame id, detector boxes with scores, and (once available) logged gimbal angles. A line-per-frame JSONL is enough; replay without the video for tracker tests and with the video for visual overlays. The existing webcam recorder stores footage with an FPS value; add a timestamp per frame.
2. **Detector output without hardware**: run YOLO or the classical detector on the recorded video once, dump boxes. Everything after that is deterministic and fast.
3. **No gimbal angles in old footage**: assume pan = tilt = 0 (static camera). To exercise the world-angle code, **inject a synthetic gimbal trajectory**: pick `pan(t)`, `tilt(t)`, shift each recorded detection by the corresponding pixel offset (`-fx*d_pan`), and check the tracker recovers the original static world positions. This is the key unit test for the world-angle claim.
4. **Tracker tests**: synthetic balloons with known truth (static, drifting, crossing, occluded, flickering detections, false positives); assert ID continuity, position RMSE, NIS around 1 per axis, birth/confirm/delete counts. Compare the hand-written filter against `filterpy`/`scipy` as an oracle in tests only (hand-written in production code per the team rule).
5. **Simulator for the loop**: a 2D plant, one instance per axis:
```
theta_dot[k+1] = clip(accel-limited toward w_cmd(k - delay_steps))   # stepper: rate command
theta[k+1]     = theta[k] + theta_dot*dt (+ backlash, quantisation to step size)
camera:  u = cx + fx*tan(az_target - theta_pan)  (+ pixel noise, +/-1 frame jitter, blur-dependent noise)
```
   with: frame period 33 ms with jitter, delay `L_cam + L_inf + L_serial`, rate and acceleration limits, a servo variant (first-order lag plus deadband plus angle quantisation), and optional gimbal-angle noise. Feed simulated boxes into the same tracker and controller code. Metrics: settling time to within 0.5 deg, overshoot, steady-state error, tracking error on a ramping target, time to fire, and fire-gate false-fire rate.
6. **Tuning workflow**: grid-search `Kp, Ki, Kd, e_db, N_fire` against the simulator for both plant types; check gain and delay margin by sweeping `L` to 2x nominal. Re-tune with the measured plant (step response: command a step, log the response) once hardware exists.
7. **Latency measurement test** (needs the camera): flash an LED or screen timer, compare frame content to timestamps; this fixes `L_cam`.

## 7. Suggested build plan

Each step lists its offline test and what it needs from the detectors.

1. **Camera model and pixel/angle utilities** (`to_world`, `from_world`, parallax and range from width). Offline test: round-trip pixel to angle to pixel within 1e-6; synthetic point at known angle; compare against `cv2.undistortPoints` once calibrated. Calibrate the C270 with chessboard images and store K and distortion; start with the 55 deg prior (f about 1411 px) until then. Detectors: none (any box centre and width).
2. **Detector replay format and recorder** (JSONL with `t_capture`, boxes `x1,y1,x2,y2,score`, optional gimbal angles). Offline test: replay reproduces timestamps and box counts, overlay video looks right. Detectors: must emit boxes with scores and the frame's capture timestamp; class label only if more than one class.
3. **Hand-written 1-axis then 4-state CV Kalman filter** with variable dt (F, Q, Joseph update). Offline test: constant-velocity synthetic data converges to truth; matches `filterpy.KalmanFilter` output to 1e-9 in a test oracle; NIS about 1 on simulated data; dt jitter and dropout cases. Detectors: none (synthetic), then box centres from recordings of a stationary balloon to measure `sigma_z`.
4. **Track manager, SORT-style**: birth, tentative to confirmed (M of N), coasting, deletion, Mahalanobis gating, exact assignment (brute-force) in angle space. Offline test: synthetic multi-balloon scenarios (crossing, near pairs 1 deg apart, flicker, false positive) with ID-switch counts; replay recorded multi-balloon footage and check track count equals balloon count. Detectors: per-frame multiple boxes, score threshold chosen so false positives are rare; stable box centres (note whether YOLO or the classical detector has more jitter, since R depends on it).
5. **World-angle conversion with gimbal angle log** (ring buffer, interpolation, motion-blur-inflated R). Offline test: the synthetic-gimbal-trajectory injection described in section 6.3; static balloons must stay static in world angles. Detectors: the same boxes; just require true capture timestamps (not processing time).
6. **Gimbal plant simulator** (stepper rate-command and servo position-command variants, delay, limits, noise). Offline test: step response matches analytic first-order or delay behaviour; limit and quantisation behaviour unit-tested. Detectors: none (the simulator renders its own boxes), so the loop can be developed before footage and before motors are decided.
7. **P controller with deadband, rate and acceleration limits, then KF-velocity feedforward**; add I with anti-windup and D-on-measurement only if the simulator shows a need. Offline test: simulator step and ramp tracking, within 0.5 deg inside the target settling time at 2x nominal delay; no windup under saturation test; no dithering inside deadband. Detectors: simulated or replayed track; needs only track state.
8. **Target selection with hysteresis** and the **fire gate**. Offline test: simulated multi-balloon scenes; no target flip-flop, no firing before N stable frames, no fire on unconfirmed tracks, no fire mid-slew. Detectors: confirmed tracks with width (for range) and score.
9. **Lead, ballistic and parallax offsets, and pop/miss confirmation.** Offline test: simulated drifting balloons with set dart speed and drop; verify hit rate vs latency and the hit/miss classifier on synthetic vanish events (true pop, occlusion, detector dropout). Detectors: box width reliable enough for range within about 10 percent; dropout statistics measured from recordings to set `T_lost`.
10. **Hardware integration**: serial protocol, measure `L_cam`, calibrate boresight point `s*`, fit dart speed and drop table, re-identify the plant from step tests, re-tune in the simulator, then closed-loop on static balloons before moving ones. Offline test: replay logged hardware sessions through the tracker and compare to simulation. Detectors: frame timestamps at the moment of capture; latency per frame if the detector reports it.

## 8. Open questions and risks

- Stepper vs servo decides whether angle feedback and sub-degree accuracy are realistic; a hobby servo has no readback and may not meet 0.5 deg, and the host's gimbal angle would then only be a commanded value.
- Camera latency and autofocus on the C270 are unmeasured; autofocus changes f and breaks calibration, so disable it.
- Parallax and ballistic drop each match the whole error budget at 3 to 6 m, so range from box width must be accurate.
- Items marked ASSUMPTION (dart speed, barrel offset, latencies, balloon drift, servo resolution) are placeholders to be measured.
