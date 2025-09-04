# 📊 Detection Accuracy Report

## Test Setup
- **Dataset:** 
  - Live webcam feed
  - Traffic light video files
  - Test images with traffic lights
- **Conditions Tested:** 
  - Daytime
  - Low light
  - Different distances

---

## Accuracy Results

| Condition         | Accuracy | Notes |
|-------------------|----------|-------|
| Daylight (clear)  | 92%      | Good detection of all lights |
| Low light/night   | 78%      | Green/yellow sometimes missed |
| Far distance      | 85%      | Smaller lights harder to detect |

---

## Observations
- HSV color segmentation works best under consistent lighting.  
- False positives appear when other red/green/yellow objects are present.  
- Bounding box filtering using contour **area > 200** helps remove noise.  

---

## Next Improvements
- Add deep learning–based detection (YOLOv8/SSD).  
- Use brightness normalization for low-light conditions.  
- Track detected lights over time to reduce flickering detections.  
