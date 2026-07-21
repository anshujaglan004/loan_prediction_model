import os
import joblib
import gradio as gr

# Load trained model
try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print("Model loading error:", e)
    model = None


def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    if model is None:
        return "❌ Model not loaded."

    try:
        input_data = [
            [
                int(no_of_dependents),
                int(education),
                int(self_employed),
                float(income_annum),
                float(loan_amount),
                int(loan_term),
                int(cibil_score),
                float(residential_assets_value),
                float(commercial_assets_value),
                float(luxury_assets_value),
                float(bank_asset_value),
            ]
        ]
    except:
        return "❌ Please enter valid values."

    # Validation
    if input_data[0][6] < 300 or input_data[0][6] > 900:
        return "❌ CIBIL Score must be between 300 and 900."

    if any(value < 0 for value in input_data[0]):
        return "❌ Negative values are not allowed."

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            return "🟢 Loan Approved"
        else:
            return "🔴 Loan Rejected"

    except Exception as e:
        return f"Prediction Error: {e}"


interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Dropdown(
            choices=[("Graduate", 1), ("Not Graduate", 0)],
            label="Education",
        ),
        gr.Dropdown(
            choices=[("Yes", 1), ("No", 0)],
            label="Self Employed",
        ),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score"),
        gr.Number(label="Residential Assets"),
        gr.Number(label="Commercial Assets"),
        gr.Number(label="Luxury Assets"),
        gr.Number(label="Bank Assets"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Loan Approval Prediction System",
    description="Enter the applicant details and click Submit.",
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
