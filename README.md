# Vertlytics: AI Jumping Coach & Injury Risk Analyzer

Vertlytics is an AI-powered tool that transforms a single video of your vertical jump into a comprehensive biomechanical analysis. Designed to empower both professional athletes and the general public, Vertlytics provides personalized insights and actionable recommendations to prevent injuries and optimize performance—all from just one video.

Access our official [Devpost submission page](https://devpost.com/software/hello-jmohbp), including a comprehensive **Demo video** along with an in-depth project breakdown and documentation.

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Features](#features)
- [Technical Overview](#technical-overview)
- [Data & Research](#data--research)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Future Directions](#future-directions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Background & Motivation

In high-impact sports and daily activities alike, improper landing techniques can lead to severe injuries such as ACL tears, chronic knee pain, and long-term joint damage. Traditionally, detailed biomechanical analysis has been confined to specialized labs with expensive equipment. Vertlytics bridges that gap by making professional-level movement analysis accessible to everyone. Using advanced computer vision, AI, and data analytics, Vertlytics captures key movement metrics from a single video to assess jump performance and injury risk.

---

## Features

- **Simple Video Upload:**  
  Users can upload a video of their vertical jump (supporting .mov and .mp4 formats). The app automatically converts, rotates, and processes the video to ensure compatibility.

- **Advanced Biomechanical Analysis:**  
  Using MediaPipe, OpenCV, and FFmpeg, Vertlytics extracts key landmarks (toes, knees, hips, ankles) and calculates critical metrics:
  - **Knee Alignment (Valgus/Varus):** Measures dynamic knee deviation.
  - **Knee Flexion Angle:** Assesses landing shock absorption.
  - **Hip Drop:** Evaluates core and gluteal control.
  - **Jump Height:** Estimates vertical jump performance.
  - **Asymmetry & Variability:** Quantifies limb imbalances and consistency.

- **Composite Risk Scoring:**  
  Metrics are compared against normative values from sports biomechanics research, and a composite risk score is computed using weighted z-scores. The score is then mapped to detailed risk categories ranging from “Optimal” to “Very High Risk.”

- **Personalized Recommendations:**  
  Based on your analysis, Vertlytics provides tailored training and corrective exercise recommendations to reduce injury risk and improve performance.

- **Interactive Visualizations:**  
  The app overlays key points and event markers (e.g., takeoff, apex, landing) directly on the video. Interactive plots display time-series data of key metrics, making it easy to interpret your performance.

---

## Technical Overview

Vertlytics is built entirely in Python, leveraging the following technologies:

- **Computer Vision:**  
  OpenCV for video processing and MediaPipe for landmark detection.
- **Video Conversion:**  
  FFmpeg is used to convert and encode videos to H264 with a compatible pixel format.
- **Web Framework:**  
  Streamlit provides a fast, interactive, and visually pleasing web interface.
- **Data Visualization:**  
  Matplotlib (with plans to integrate interactive libraries like Plotly) is used for generating detailed plots and charts.
- **Machine Learning & Data Analytics:**  
  Our composite risk scoring model normalizes user data against normative datasets (e.g., CMU Motion Capture, Human3.6M) and applies weighted statistical models to generate personalized feedback.

---

## Data & Research

Vertlytics is built on a strong data-driven foundation:

- **Datasets:**  
  - **CMU Motion Capture Database:** Used to understand joint kinematics in dynamic movements.
  - **Human3.6M:** Provided extensive 3D motion data, essential for modeling realistic human movement.
  - **Kaggle Datasets on Sports Performance:** Informed our understanding of typical jump heights and landing mechanics in athletes.

- **Key Research:**  
  - **ACL Injury Risk:** Studies by Hewett et al. (2005) highlight the role of excessive knee valgus in ACL injuries.
  - **Landing Mechanics:** Research by Myer et al. (2005) emphasizes the importance of adequate knee flexion to reduce injury risk.
  - **Neuromuscular Control:** Recent literature confirms that asymmetry and variability in joint kinematics are strong predictors of injury risk.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/my_jump_analysis_project.git
   cd my_jump_analysis_project
   
