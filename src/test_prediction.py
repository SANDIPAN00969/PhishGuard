# ============================================
#   PHISHGUARD — Local Test Script
#   Run this file to test the pipeline
# ============================================

import sys
import os

# Add src folder to path so we can import prediction_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_pipeline import load_models, predict_email

def run_test():

    # Step 1: Load all models
    print("=" * 55)
    print("   PhishGuard — Loading Models")
    print("=" * 55)
    print("⏳ Loading all models... please wait")

    models = load_models()
    print("✅ All models loaded successfully!\n")

    # Step 2: Define test emails
    test_emails = [

        # Phishing email with URL
        {
            "description": "Phishing email with suspicious URL",
            "email": """
            Dear Valued Customer,
            Your account has been suspended due to unusual activity.
            You must verify your account immediately or it will be
            permanently deleted within 24 hours.
            Please click here to confirm your details:
            http://paypal-secure-verify-login.com/confirm?id=abc123
            Act now! This is urgent!
            """
        },

        # Phishing email without URL
        {
            "description": "Phishing email without URL",
            "email": """
            URGENT NOTICE: Congratulations! You have won a prize
            of $10,000 in our annual lottery. To claim your prize
            immediately, please send your bank account details,
            social security number and password to confirm
            your identity. Limited time offer expires soon!
            Act now or lose your prize forever!
            """
        },

        # Legitimate email with URL
        {
            "description": "Legitimate email with URL",
            "email": """
            Hi team, please find the quarterly report attached.
            You can also access the full report on our company
            portal at https://www.companyportal.com/reports/q3
            The meeting to discuss results is scheduled for
            Friday at 2pm in conference room B.
            Best regards, John
            """
        },

        # Legitimate email without URL
        {
            "description": "Legitimate email without URL",
            "email": """
            Hi Sarah, hope you are doing well.
            Just wanted to follow up on the project timeline
            we discussed last week. Could you please send me
            the updated schedule when you get a chance?
            No rush, whenever is convenient for you.
            Thanks, Mike
            """
        }
    ]

    # Step 3: Run predictions
    print("=" * 55)
    print("   PhishGuard — Running Predictions")
    print("=" * 55)

    for i, sample in enumerate(test_emails):
        print(f"\n📧 Test Email {i+1}: {sample['description']}")
        print("-" * 55)

        # Run prediction
        result = predict_email(sample['email'], models)

        # Display result
        emoji = "⚠️ " if result['result'] == "PHISHING" else "✅"
        print(f"  Result          : {emoji} {result['result']}")
        print(f"  Confidence      : {result['confidence']}")
        print(f"  XGBoost Score   : {result['xgb_text_score']}%")
        print(f"  DistilBERT Score: {result['bert_score']}%")
        print(f"  URL Score       : {result['url_score']}")
        print(f"  Final Score     : {result['final_score']}%")
        print("-" * 55)

    print("\n✅ All predictions complete!")
    print("🛡️  PhishGuard is working perfectly!")

# Run the test
if __name__ == "__main__":
    run_test()