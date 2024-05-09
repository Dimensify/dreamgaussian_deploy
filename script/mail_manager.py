import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(send_email, userid, status):
    if send_email and userid:
        # Email configuration
        smtp_host = 'live.smtp.mailtrap.io'
        smtp_port = 587
        smtp_username = 'api'
        smtp_password = '2aac3c775215e012a2598effc0d4c6dc'
        sender_email = 'dimensify3D@dimensify.ai'
        recipient_email = userid

        # Email content
        if status and status['result'] == 'SUCCESS':
            subject = 'Your Dimensify 3D model is ready'
            body = """
                <!DOCTYPE html>
                <html>
                <body>
                <p>Hello there,</p>
                <p>Dimensify 3D model is ready, please click the link below:</p>
                <p>https://dev.dimensify.ai/user/my-space</p>
                <br><br><br><br>
                <p>Your Dimensify Team</p>
                </body>
                </html>
            """
        else:
            subject = 'Dimensify 3D model creation failed'
            body = """
                <!DOCTYPE html>
                <html>
                <body>
                <p>Hello there,</p>
                <p>Dimensify 3D model creation failed, please try again!</p>
                <p>https://dev.dimensify.ai/account</p>
                <br><br><br><br>
                <p>Your Dimensify Team</p>
                </body>
                </html>
            """

        # Set up SMTP connection
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        # Create email message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject

        # Attach HTML content
        message.attach(MIMEText(body, 'html'))

        # Send email
        server.send_message(message)

        # Close SMTP connection
        server.quit()

        return "Message is sent"
    else:
        return "Send email option or user ID is missing"