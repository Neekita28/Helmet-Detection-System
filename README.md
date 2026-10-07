# 👷 Helmet Detection System: Real-Time Worker Head Detection & Occupancy Monitor

An edge-ready Computer Vision pipeline built with **YOLOv8** and **Streamlit** to detect on-site personnel, track headcount density, and monitor worker presence across industrial and construction environments in real time.


## 📌 Project Overview

Monitoring high-risk construction zones and manufacturing floors requires strict auditing of personnel presence and zone density. Standard generic object detectors often struggle in cluttered industrial settings where full bodies may be occluded by scaffolding, machinery, or materials.

**SmartSite-Vision** addresses this by localizing the primary anatomical anchor point—the **worker's head**. This enables:
* Robust worker localization even when torso/limbs are occluded.
* Real-time headcount estimation and occupancy analytics.
* A baseline localization architecture designed for downstream PPE compliance pipelines (e.g., hard-hat/face-shield classification).

---

## 🚀 Key Features

* **High-Throughput Localization:** Powered by a customized YOLOv8 architecture optimized for rapid inference on standard edge hardware and CPUs.
* **Interactive Dashboard:** Streamlit-powered GUI supporting single-frame image auditing and continuous video processing.
* **Real-Time Threshold Tuning:** Dynamic confidence slider allowing operators to filter low-confidence edge detections on the fly.
* **Occupancy Analytics:** Immediate visual feedback with automatic headcount counters and detection summaries.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Deep Learning Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Computer Vision Library:** OpenCV (`cv2`)
* **Dashboard / UI:** [Streamlit](https://streamlit.io/)
* **Numerical Processing:** NumPy, Pillow

---

## 📁 Repository Structure

```text
├── models/
│   └── best.pt               # Trained YOLOv8 model weights
├── sample_data/              # Sample test images and video clips
├── app.py                    # Streamlit deployment application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
