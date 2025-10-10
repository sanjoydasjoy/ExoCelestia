# 🚀 Authentication Quick Reference Card

## ⚡ Installation (Copy & Paste)

```powershell
# Install dependencies
cd backend
pip install python-jose[cryptography] passlib[bcrypt] pydantic-settings
cd ..

# Start backend (Terminal 1)
cd backend
uvicorn app.main:app --reload --port 8000

# Start frontend (Terminal 2)
cd frontend
npm run dev

# Open browser
start http://localhost:3000
```

---

## 📍 Files Created/Modified

### ⭐ New Files
```
backend/app/auth.py              # Auth logic & JWT
backend/app/auth_models.py       # User models
backend/app/api/auth.py          # API endpoints
frontend/contexts/AuthContext.tsx # Auth state
frontend/components/AuthModal.tsx # Login/Signup UI
```

### ✏️ Modified Files
```
backend/requirements.txt         # +3 packages
backend/app/main.py             # +auth router
frontend/pages/_app.tsx         # +AuthProvider
frontend/pages/index.tsx        # +auth UI
```

---

## 🎯 Quick Usage

### Frontend - Use Auth Hook
```typescript
import { useAuth } from '../contexts/AuthContext';

const { user, token, login, signup, logout } = useAuth();

// Check if logged in
if (user) {
  console.log(`Logged in as ${user.email}`);
}

// Login
await login('user@example.com', 'Password123');

// Signup
await signup('new@example.com', 'Password123', 'John Doe');

// Logout
logout();
```

### Backend - Protect Routes
```python
from app.auth import get_current_active_user
from app.auth_models import User
from fastapi import Depends

@router.get("/protected")
async def my_route(user: User = Depends(get_current_active_user)):
    return {"message": f"Hello {user.email}"}
```

### Frontend - Authenticated Requests
```typescript
const { token } = useAuth();

fetch('/api/endpoint', {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

---

## 🔌 API Endpoints

### POST /api/auth/signup
```json
{
  "email": "astronaut@nasa.gov",
  "password": "SecurePass123",
  "full_name": "John Doe"  // optional
}
```

### POST /api/auth/login
```json
{
  "email": "astronaut@nasa.gov",
  "password": "SecurePass123"
}
```

### GET /api/auth/me
```
Headers: Authorization: Bearer <token>
```

---

## 🎨 UI Components

### Show Auth Modal
```typescript
const [showModal, setShowModal] = useState(false);
const [mode, setMode] = useState<'login' | 'signup'>('login');

<AuthModal 
  isOpen={showModal}
  onClose={() => setShowModal(false)}
  initialMode={mode}
/>
```

### Check Auth State
```typescript
const { user, isLoading } = useAuth();

if (isLoading) return <Spinner />;
if (user) return <Dashboard />;
return <LoginPrompt />;
```

---

## 🔒 Password Rules

- Minimum 8 characters
- At least 1 uppercase letter (A-Z)
- At least 1 lowercase letter (a-z)
- At least 1 number (0-9)

Example: `SecurePass123` ✅

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module 'jose' not found | Run: `pip install python-jose[cryptography]` |
| CORS error | Check backend is on port 8000 |
| Token not persisting | Check browser localStorage |
| Cannot login | Verify credentials, check backend logs |
| Modal not appearing | Check AuthModal is imported |

---

## 📦 Dependencies Added

```txt
python-jose[cryptography]==3.3.0  # JWT tokens
passlib[bcrypt]==1.7.4            # Password hashing  
pydantic-settings==2.1.0          # Settings management
```

---

## ✅ Testing Checklist

- [ ] Click "Sign Up" - modal appears
- [ ] Create account - auto logged in
- [ ] See profile avatar in nav
- [ ] Click avatar - dropdown shows
- [ ] Click "Sign Out" - logged out
- [ ] Click "Login" - modal appears
- [ ] Login with credentials - success
- [ ] Refresh page - still logged in

---

## 📖 Full Documentation

- **Setup Guide**: `QUICK_START.md`
- **Complete Docs**: `AUTH_IMPLEMENTATION.md`
- **Summary**: `AUTHENTICATION_COMPLETE.md`

---

**Ready to go! 🎉**

