import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_rf = joblib.load("loan_prediction_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    deployed_rf = None


# ==========================================================
# Prediction Function
# ==========================================================
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

    values = [
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
    ]

    # Check for empty fields
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all input fields."

    try:
        no_of_dependents = int(no_of_dependents)
        education = int(education)
        self_employed = int(self_employed)
        income_annum = float(income_annum)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)
        cibil_score = int(cibil_score)
        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)

    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."

    # Validation
    numeric_values = [
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
    ]

    if any(v < 0 for v in numeric_values):
        return "❌ Negative values are not allowed."

    if not (300 <= cibil_score <= 900):
        return "❌ CIBIL Score must be between 300 and 900."

    if no_of_dependents > 20:
        return "❌ Number of dependents cannot exceed 20."

    if deployed_rf is None:
        return "❌ Model failed to load."

    try:
        input_data = [[
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
        ]]

        prediction = deployed_rf.predict(input_data)

        if prediction[0] == 1:
            return (
                "🟢 Prediction Result\n\n"
                "Loan Status: APPROVED\n\n"
                "Congratulations! The applicant is likely eligible for loan approval."
            )
        else:
            return (
                "🔴 Prediction Result\n\n"
                "Loan Status: REJECTED\n\n"
                "The applicant is unlikely to qualify for loan approval."
            )

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {e}"


# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 🏦 Loan Approval Prediction System

This application predicts whether a loan application is likely to be **Approved** or **Rejected** using a trained **Random Forest Machine Learning Model**.

Enter all applicant details below and click **Submit** to get the prediction.
"""


# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Dropdown(
            choices=[("Graduate", 1), ("Not Graduate", 0)],
            label="Education Status",
        ),
        gr.Dropdown(
            choices=[("Yes", 1), ("No", 0)],
            label="Self Employed",
        ),
        gr.Number(label="Annual Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="CIBIL Score (300 - 900)"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value"),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=6),
    title="🏦 Loan Approval Prediction System",
    description=DESCRIPTION,
)


# ==========================================================
# Launch App
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
