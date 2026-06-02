# Authentication Implementation Summary for Frontend Team

## Overview
Authentication has been added to the backend API using JWT (JSON Web Tokens). The Next.js frontend now needs to handle user authentication by obtaining tokens from the backend and including them in API requests.

## Endpoints

### 1. User Registration
- **URL**: `POST /api/auth/register`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe" (optional)
  }
  ```
- **Response**:
  ```json
  {
    "id": "user-uuid",
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true
  }
  ```

### 2. User Login
- **URL**: `POST /api/auth/token`
- **Content-Type**: `application/x-www-form-urlencoded` (OAuth2 standard)
- **Form Data**:
  - `username`: user's email
  - `password`: user's password
- **Response**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
  ```

## Using the API with Authentication

### Storing the Token
After login, store the access token securely (e.g., in httpOnly cookie, localStorage, or sessionStorage).

### Making Authenticated Requests
Include the token in the Authorization header:
```
Authorization: Bearer <access_token>
```

### Protected Endpoints
All existing API endpoints now require authentication:
- `POST /api/chat/message` - Send a chat message
- `GET /api/chat/session/{session_id}` - Get session messages
- `POST /api/chat/reset/{session_id}` - Reset a session
- `POST /api/learn/message` - Learning flow endpoint
- `POST /api/notion/create-topic` - Create Notion topic

## Example Flow (Frontend)

```javascript
// 1. Login
const loginResponse = await fetch('/api/auth/token', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    username: email,
    password: password
  })
});
const { access_token } = await loginResponse.json();

// 2. Store token (example using localStorage - consider more secure options)
localStorage.setItem('access_token', access_token);

// 3. Make authenticated request
const chatResponse = await fetch('/api/chat/message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    message: "Hello, I want to learn React"
  })
});

// 4. Include session-id header as before (now associated with authenticated user)
```

## Security Considerations
1. **Token Storage**: Avoid storing tokens in localStorage for production applications due to XSS risks. Consider:
   - httpOnly cookies (most secure)
   - Session storage
   - Secure storage mechanisms

2. **Token Expiration**: Tokens expire after 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES` in .env)

3. **Refresh Tokens**: Not implemented in this version. For production, consider implementing refresh token flow.

4. **HTTPS**: Ensure your Next.js app is served over HTTPS in production to protect tokens in transit.

## Backend Changes Summary
- Added user registration and login endpoints
- Implemented password hashing with bcrypt
- Added JWT token generation and validation
- Modified session handling to associate sessions with authenticated users
- Protected all API routes with authentication dependencies
- Added CORS headers to allow Authorization header

## Testing the API
You can test the endpoints using tools like curl or Postman:

```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","full_name":"Test User"}'

# Login
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=test123"

# Use token in request
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
  -H "session-id: your-session-id" \
  -d '{"message":"Hello"}'
```

## Next Steps for Frontend
1. Implement login/register forms in Next.js
2. Handle token storage and retrieval
3. Add Authorization header to all API requests
4. Handle token expiration and refresh (if needed)
5. Update session management to work with authenticated users