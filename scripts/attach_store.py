"""DEV ONLY: attach an existing store to a user. Usage: python -m scripts.attach_store EMAIL STORE_ID"""

import sys

from app.auth.repository import add_store_member, get_user_by_email, store_has_members

email, store_id = sys.argv[1], sys.argv[2]
user = get_user_by_email(email)
if user is None:
    sys.exit(f"No user with email {email}")
if store_has_members(store_id):
    sys.exit(f"{store_id} already has an owner. Refusing.")
add_store_member(user["user_id"], store_id, "owner")
print(f"Attached {store_id} to {email}")