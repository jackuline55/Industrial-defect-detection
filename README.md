# **Industrial-defect-detection**

# **Purpose**
The primary objective of this project is to detect industrial defects in visual data such as product surfaces or components using basic image processing techniques. The detection is based on identifying edges, contours, and potential anomalies like missing parts or unwanted features, which could indicate a defect. Additionally, it demonstrates how facial detection and color segmentation techniques can be repurposed for object localization and defect identification in industrial applications.

# **Technology Used**
Programming Language: Python

Image Processing Library: OpenCV

Environment: Google Colab (for execution and visualization)

Machine Learning: Haar Cascade Classifier (for face detection, adaptable to object detection)

Techniques Used:

Grayscale conversion

Edge detection (Canny)

Contour detection

Object detection (Haar cascade)

Blurring techniques (Average, Gaussian, Median)

Color-based segmentation using HSV color space

Real-time video capture via webcam (for live monitoring)

# **Usage & Benefits**
This system performs a series of image processing tasks that simulate an industrial defect detection workflow. It includes image pre-processing, defect highlighting through edge and contour detection, color analysis, and real-time monitoring. The approach provides foundational capabilities that can be scaled for automated quality control in manufacturing environments.

**Benefits of Industrial Defect Detection Systems:**

**Increased Product Quality**: Early detection of visual defects ensures that only high-quality products reach customers, enhancing brand reputation.
**Reduced Human Error**: Automating inspection reduces reliance on manual checks, which can be inconsistent and prone to oversight.
**Improved Production Speed**: Real-time monitoring and automated inspections streamline the quality control process without slowing down production.
**Worker Safety**: Minimizes the need for workers to manually inspect potentially hazardous machinery or environments.

# **Conclusion**
This project demonstrates the application of fundamental computer vision techniques for defect detection in industrial settings. By leveraging OpenCV functions such as edge detection, contour extraction, object recognition, and real-time video processing, the system provides a foundational framework for automated visual inspection. These tools are effective for identifying surface-level defects, missing parts, or color inconsistencies in manufactured products, and the code can be extended with more advanced techniques such as deep learning for enhanced accuracy and reliability in complex environments.



