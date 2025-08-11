# Authentication API
Latest backend workflow is documented in the repo root at ../Latest-project-workflow.md.
A comprehensive user authentication system built with FastAPI and MongoDB, featuring JWT tokens, password hashing, rate limiting, and role-based access control foundations.

## Features

### Core Authentication
- ✅ User registration with email validation
- ✅ User login with JWT token generation
- ✅ Token refresh mechanism
- ✅ User logout with token blacklisting
- ✅ Password change for authenticated users

### Security Features
- ✅ Secure password hashing with bcrypt
- ✅ JWT tokens with proper expiration
- ✅ Token blacklisting for logout
- ✅ Input validation with Pydantic models
- ✅ CORS configuration
- ✅ Comprehensive error handling

### User Management
- ✅ User profile management
- ✅ Public user profiles (limited information)
- ✅ Extensible user schema for future features
- ✅ Role-based access control foundations

### Technical Features
- ✅ FastAPI with automatic OpenAPI documentation
- ✅ MongoDB with Motor (async driver)
- ✅ Proper database indexing
- ✅ Environment-based configuration
- ✅ Comprehensive logging
- ✅ Unit and integration tests
- ✅ Production-ready error handling

## Quick Start

### Prerequisites
- Python 3.11+
- MongoDB
- uv (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start MongoDB (if running locally):
```bash
mongod
```

5. Run the application:
```bash
uv run python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the application is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and adjust the values:

### Required Settings
- `SECRET_KEY`: JWT secret key (minimum 32 characters)
- `MONGODB_URL`: MongoDB connection string
- `DATABASE_NAME`: MongoDB database name

### Optional Settings
- `DEBUG`: Enable debug mode (default: false)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Access token expiry (default: 30)
- `REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token expiry (default: 7)

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/change-password` - Change password (authenticated)

### User Management
- `GET /api/v1/users/me` - Get current user profile
- `PUT /api/v1/users/me` - Update current user profile
- `GET /api/v1/users/{user_id}` - Get public user profile

### Health Check
- `GET /health` - Application health check

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app

# Run specific test file
uv run pytest tests/test_auth.py

# Run with verbose output
uv run pytest -v
```

## Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  email: String (unique),
  hashed_password: String,
  is_active: Boolean,
  is_verified: Boolean,
  is_superuser: Boolean,
  profile: {
    first_name: String,
    last_name: String,
    phone_number: String,
    bio: String,
    preferences: Object
  },
  roles: Array,
  created_at: Date,
  updated_at: Date,
  last_login: Date,
  failed_login_attempts: Number,
  locked_until: Date,
  password_changed_at: Date,
  metadata: Object
}
```

### Token Blacklist Collection
```javascript
{
  _id: ObjectId,
  token: String (unique),
  user_id: ObjectId,
  expires_at: Date,
  blacklisted_at: Date,
  reason: String
}
```

# Password reset tokens collection removed

## Security Considerations

### Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

### Account Security
- JWT tokens have configurable expiry times
- Secure password requirements with validation

## Development

### Project Structure
```
backend/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core configuration
│   ├── db/            # Database connection
│   ├── middleware/    # Authentication middleware
│   ├── models/        # Database models
│   ├── schemas/       # Pydantic schemas
│   └── services/      # Business logic
├── tests/             # Test files
├── main.py           # Application entry point
└── README.md         # This file
```

### Adding New Features

1. **Database Models**: Add to `app/models/`
2. **API Schemas**: Add to `app/schemas/`
3. **Business Logic**: Add to `app/services/`
4. **API Endpoints**: Add to `app/api/`
5. **Tests**: Add to `tests/`

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to functions and classes
- Write tests for new functionality

## Deployment

### Environment Variables
Ensure all required environment variables are set in production:
- Use a strong, unique `SECRET_KEY`
- Set `DEBUG=false`
- Configure proper MongoDB connection
- Set up CORS origins for your frontend

### Security Checklist
- [ ] Strong secret key configured
- [ ] Debug mode disabled
- [ ] HTTPS enabled
- [ ] Database secured
- [ ] Rate limiting configured
- [ ] Logging configured
- [ ] Error handling tested

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License.