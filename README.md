# HealthCare ChatBot

A machine learning-powered chatbot for preliminary disease diagnosis based on user-reported symptoms. This interactive tool helps users identify potential health issues by guiding them through symptom selection and providing disease predictions, descriptions, and precautions.

## Features

- **Interactive Symptom Assessment**: Users select the affected body part, choose relevant symptoms from a curated list, and specify symptom duration.
- **Disease Prediction**: Utilizes a Decision Tree Classifier trained on medical symptom data to predict possible diseases.
- **Comprehensive Information**: Provides disease descriptions, recommended precautions, and severity-based advice.
- **User-Friendly Web Interface**: Built with Streamlit for an intuitive web-based experience.
- **Data-Driven Insights**: Leverages symptom severity scores and duration to offer tailored health advice.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/acedev97/health_consultant.git
   cd health_consultant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install pandas scikit-learn streamlit numpy
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run gui.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Follow the interactive prompts:
   - Enter your name
   - Select the affected body part
   - Choose relevant symptoms from the displayed list
   - Specify how long you've experienced the symptoms (in days)
   - Click "Predict Disease" to get results

## Data

The project uses the following datasets (located in the `Data/` and `MasterData/` directories):
- `Training.csv` & `Testing.csv`: Symptom data for training and testing the ML model
- `symptom_Description.csv`: Descriptions of various symptoms
- `symptom_precaution.csv`: Precautions for different diseases
- `Symptom_severity.csv`: Severity scores for symptoms

## Project Structure

```
health_consultant/
├── gui.py                 # Main Streamlit application
├── Data/
│   ├── Training.csv
│   ├── Testing.csv
│   └── dataset.csv
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom_severity.csv
├── README.md
└── requirements.txt       # (Create this file with dependencies)
```

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- streamlit
- numpy

## How It Works

1. **Body Part Selection**: Users choose from predefined body parts (head, chest, stomach, skin, limbs, urinary, general).
2. **Symptom Filtering**: Based on the selected body part, relevant symptoms are displayed for user selection.
3. **Duration Input**: Users provide the duration of symptoms to assess severity.
4. **Prediction**: The ML model analyzes selected symptoms to predict the most likely disease.
5. **Results**: Displays predicted disease, description, precautions, and additional advice based on symptom severity and duration.

## Disclaimer

This chatbot is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
