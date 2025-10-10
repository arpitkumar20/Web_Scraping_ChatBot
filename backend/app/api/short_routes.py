from flask import Blueprint, redirect

# Create blueprint
shorter_routes = Blueprint("routes", __name__)

# Redirect route
@shorter_routes.route("/booking", methods=['GET'])
def booking_redirect():
    print("Redirecting to booking page...")
    booking_url = "https://www.peerlesshospital.com/doctor.php"
    return redirect(booking_url, code=302)


# Redirect route
@shorter_routes.route("/cancel", methods=['GET'])
def cancel_redirect():
    print("Redirecting to cancle page...")
    booking_url = "https://www.peerlesshospital.com/doctor.php"
    return redirect(booking_url, code=302)