# Colour spaces (HSV)
Saturation and Hue can be used to differentiate similar objects. We can config for colour essentially. Hue range is 0-179.

HSV is essentially combinging what a colour is so to say instead of BRG. It allows us to detect very specific colours with specific HSV values.

Low saturation makes hue unreliable. 

# Thresholding and masks
Put an image (Cyan notebook) infront of the camera and wrote down the range of HSV (Hue, Saturation, Value). I then made a mask for those values, showing them in white. Then I made everything else black and showed only that one colour (blended_mask)

My values for cyan include the following:
lower_bound = np.array([85, 130, 130])
upper_bound = np.array([100, 255, 255])

# Contours
Uses binary imaging to draw shapes around white objects in a black background.

Calling cv2.findContours() returns the contours as a list, but also the contours inside contours. 

I am drawing a rectangular bounding box since they are shareable with a YOLO model, and I can calculate the centre easily. Also allows me to use width/height to help filter balloons shape. 

# Weaknesses and limits
Lighting, glare, similar coloured backgrounds, skin tone, and multiple contours on a mask are all potential issues. 

From testing (On my notebook) I was getting at times 30+ contours, and in the event that a balloon is far away from the camera and some similar colour within the bound is detected, it could become the largest area and thus false posative.