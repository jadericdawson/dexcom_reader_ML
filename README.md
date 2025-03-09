# Dexcom Reader

A Dexcom glucose monitoring and alerting system with multiple time-scale views, advanced trendline analysis, machine learning-based predictions, and optional Home Assistant/Nabu Casa integration.

This tool retrieves glucose readings from Dexcom, performs real-time trendline fitting using linear regression, and utilizes a Transformer-based machine learning model to predict future blood glucose (BG) values. Data is displayed in a user-friendly Tkinter GUI, with optional alerts sent to Home Assistant when BG is outside desired thresholds.

**Disclaimer:** This software is for informational and demonstration purposes only. It is not a medical device. Always consult a medical professional for any health or treatment decisions.

----------

## Features

-   **Multiple Time-Scale Plots:** 1-hour, 3-hour, 6-hour, 12-hour, 24-hour, or the full history (“Max”).
-   **Trendline Calculation:** Uses a simple linear regression on recent data to estimate future BG values.
-   **Machine Learning Predictions:** Advanced predictions using a Transformer-based model trained on historical glucose data.
-   **Tkinter GUI:** A graphical display showing current BG, trend arrows, predictive insights, and historical charts.
-   **Home Assistant Webhook Alerts:** Optionally send notifications to a Home Assistant instance via Nabu Casa when BG is out of range.

----------

## Requirements

-   **Python 3.7+**
-   **Dexcom Account:** Valid username & password to access Dexcom data via `pydexcom`.
-   _(Optional)_ **Home Assistant + Nabu Casa:** If you want to trigger alerts in your Home Assistant setup.

Dependencies include:

-   `matplotlib`
-   `numpy`
-   `pandas`
-   `torch`
-   `scikit-learn`
-   `pydexcom`
-   `pytz`
-   `python-dotenv`
-   `requests`

For details, see `pyproject.toml` or `requirements.txt`.

----------

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jadericdawson/dexcom_reader.git
cd dexcom_reader

```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .\.venv\Scripts\activate on Windows PowerShell

```

### 3. Install Dependencies

```bash
pip install .

```

----------

## Environment Variables with `.env`

This project uses `python-dotenv` to load Dexcom credentials from a local `.env` file.

Update `.env`:

```env
DEXCOM_USER="ENTER_YOUR_DEXCOM_USERNAME"
DEXCOM_PASS="ENTER_YOUR_DEXCOM_PASSWORD"

```

Ensure `.env` is ignored by version control.

----------

## Home Assistant & Nabu Casa Setup

### 1. Sign Up for Nabu Casa

Subscribe to Nabu Casa for remote webhook capabilities (~$6.50/month).

### 2. Create Webhook in Home Assistant

Follow setup instructions: [Home Assistant Installation](https://www.home-assistant.io/installation/)

Webhook URL example:

```perl
https://<your-nabu-casa-url>/api/webhook/glucose_alert

```

### 3. Update Webhook URL in Code

```python
webhook_url = "https://<your-nabu-casa-url>/api/webhook/glucose_alert"

```

----------

## Usage

Run the application:

```bash
source .venv/bin/activate
python dexcom_reader_predict.py

```

The Tkinter GUI displays:

-   Checkbox to toggle trend arrows
-   Current BG and trend description
-   Expected BG in 20 minutes
-   Historical BG graphs

Alerts via Home Assistant if configured.

----------

## Troubleshooting

-   **No Dexcom readings?** Check Dexcom credentials.
-   **Home Assistant alerts not firing?** Verify webhook URL.
-   **Installation issues?** Ensure updated Python and dependencies.

----------

## Project Structure

```bash
DEXCOM_READER/
├── .venv/                     # Virtual environment
├── advanced_model.pt          # PyTorch Transformer model
├── advanced_scaler.pkl        # Data scaler for ML model
├── dexcom_reader_predict.py   # Main prediction GUI
├── glucose_monitor.log        # Application logs
├── glucose_readings.csv       # Historical glucose data
├── glucose_readings.py        # Data management script
├── home_assistant.yaml        # Home Assistant configuration
├── model_trainer.py           # Script to train ML model
├── pyproject.toml             # Project metadata
├── README.md                  # Documentation
├── requirements.txt           # Dependency list
└── .env                       # Environment credentials

```

----------

## Contributing

Pull requests are welcome! Discuss major changes via GitHub issues first.

----------

## License

Licensed under the MIT License. See LICENSE for details.

Enjoy your Dexcom monitoring experience!