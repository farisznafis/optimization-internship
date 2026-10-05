# import library
import os
from . import helper
from dotenv import load_dotenv

try:
    # Load environment variables from .env file
    load_dotenv()

    # Get License Information from Environment Variables
    wls_access_id = os.getenv("WLSACCESSID")
    wls_secret = os.getenv("WLSSECRET")
    license_id = os.getenv("LICENSEID")

    if wls_access_id is None or wls_secret is None or license_id is None:
        license_params = {}
    else:
        license_params = {
            "WLSACCESSID": wls_access_id,
            "WLSSECRET": wls_secret,
            "LICENSEID": int(license_id),
        }

    # File Paths from Environment Variables
    employee_path = os.getenv("EMPLOYEE_PATH", "./data/employees_data.csv")
    task_path = os.getenv("TASK_PATH", "./data/tasks_data.csv")

    discord_wh_url = os.getenv("DISCORD_URL")

    helper.discord_status = False if discord_wh_url is None else helper.discord_status

except Exception as e:
    license_params = {}
    employee_path = "./data/employees_data.csv"
    task_path = "./data/tasks_data.csv"

    msg = f"Error in get the Environment Variables: {e}"
    helper.show(msg, helper.discord_status)
