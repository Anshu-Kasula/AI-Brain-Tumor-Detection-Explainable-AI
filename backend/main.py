from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# ==============================
# 🔐 JWT CONFIG
# ==============================

import os

SECRET_KEY = os.getenv("SECRET_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ==============================
# 🔐 PASSWORD HASHING
# ==============================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==============================
# 🗄 MONGODB CONNECTION
# ==============================

client = MongoClient("mongodb://localhost:27017")
db = client["AI_BrainMRI_Clinical_DB"]

users_collection = db["users"]
patients_collection = db["patients"]

# ==============================
# 📦 MODELS
# ==============================

class User(BaseModel):
    username: str
    password: str
    role: str


class PatientData(BaseModel):
    patient_name: str
    age: int
    gender: str
    phone: str
    prediction: str
    tumor_probability: float
    trust_score: float


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


class AdminResetPassword(BaseModel):
    username: str
    new_password: str


# ==============================
# 🔐 TOKEN CREATION
# ==============================

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ==============================
# 🔐 VERIFY TOKEN
# ==============================

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "username": payload.get("sub"),
            "role": payload.get("role")
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ==============================
# 👤 REGISTER (BOOTSTRAP SAFE)
# ==============================

@app.post("/register")
def register(user: User, request: Request):

    admin_exists = users_collection.find_one({"role": "admin"})

    # First admin creation (no token required)
    if not admin_exists:
        if user.role != "admin":
            raise HTTPException(status_code=400, detail="First user must be admin")

        hashed_password = pwd_context.hash(user.password)

        users_collection.insert_one({
            "username": user.username,
            "password": hashed_password,
            "role": user.role
        })

        return {"message": "First admin created successfully"}

    # After admin exists → require admin token
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admins only")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    existing = users_collection.find_one({"username": user.username})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = pwd_context.hash(user.password)

    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password,
        "role": user.role
    })

    return {"message": "User created successfully"}


# ==============================
# 🔐 LOGIN
# ==============================

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):

    user = users_collection.find_one({"username": form_data.username})

    # 🔐 SECURE CHECK
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not pwd_context.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# ==============================
# 🔐 CHANGE PASSWORD (SELF)
# ==============================

@app.post("/change_password")
def change_password(
    data: PasswordChange,
    current_user: dict = Depends(verify_token)
):
    user = users_collection.find_one({"username": current_user["username"]})

    if not pwd_context.verify(data.old_password, user["password"]):
        raise HTTPException(status_code=400, detail="Old password incorrect")

    new_hashed = pwd_context.hash(data.new_password)

    users_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {"password": new_hashed}}
    )

    return {"message": "Password updated successfully"}


# ==============================
# 👑 ADMIN RESET PASSWORD
# ==============================

@app.post("/admin_reset_password")
def admin_reset_password(
    data: AdminResetPassword,
    current_user: dict = Depends(verify_token)
):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    new_hashed = pwd_context.hash(data.new_password)

    users_collection.update_one(
        {"username": data.username},
        {"$set": {"password": new_hashed}}
    )

    return {"message": "Password reset successfully"}


# ==============================
# 💾 SAVE PATIENT
# ==============================

@app.post("/save_patient")
def save_patient(
    data: PatientData,
    current_user: dict = Depends(verify_token)
):
    if current_user["role"] not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Access denied")

    record = data.dict()
    record["doctor"] = current_user["username"]
    record["timestamp"] = datetime.utcnow()

    patients_collection.insert_one(record)

    return {"message": "Patient saved successfully"}


# ==============================
# 👨‍⚕️ DOCTOR VIEW OWN PATIENTS
# ==============================

@app.get("/my_patients")
def get_my_patients(current_user: dict = Depends(verify_token)):

    patients = list(
        patients_collection.find(
            {"doctor": current_user["username"]},
            {"_id": 0}
        )
    )

    return patients


# ==============================
# 👑 ADMIN VIEW ALL PATIENTS
# ==============================

@app.get("/all_patients")
def get_all_patients(current_user: dict = Depends(verify_token)):

    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    patients = list(patients_collection.find({}, {"_id": 0}))

    return patients
