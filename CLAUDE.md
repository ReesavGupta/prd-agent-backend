# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AI PRD Generator** - a full-stack application that transforms product ideas into comprehensive Product Requirements Documents (PRDs) using AI. The system consists of a React frontend and FastAPI backend with MongoDB for data persistence.

### Architecture
- **Frontend**: React 19 + TypeScript + Vite (port 5173)
- **Backend**: FastAPI + Motor (async MongoDB driver) + JWT authentication (port 8000)
- **Database**: MongoDB
- **Package Management**: Bun (frontend), uv (backend)

## Development Commands

### Frontend (in `frontend/` directory)
```bash
# Development server
bun run dev

# Build for production  
bun run build

# Lint code
bun run lint

# Preview production build
bun run preview
```

### Backend (in `backend/` directory)
```bash
# Install dependencies
uv sync

# Run development server
uv run python main.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=app

# Run specific test file
uv run pytest tests/test_auth.py
```

## Code Architecture

### Backend Structure
```
backend/
├── app/
│   ├── api/           # API route handlers (auth.py, users.py)
│   ├── core/          # Configuration and settings
│   ├── db/            # Database connection and setup
│   ├── middleware/    # Authentication middleware
│   ├── models/        # Pydantic/MongoDB models
│   ├── schemas/       # Request/response schemas
│   └── services/      # Business logic (auth_service.py, security.py, user_service.py)
├── tests/             # Test files
└── main.py           # FastAPI application entry point
```

### Key Backend Patterns
- **Authentication**: JWT tokens with refresh mechanism, token blacklisting for logout
- **Database**: Motor async MongoDB driver with proper indexing
- **Configuration**: Pydantic Settings with environment variable support
- **Security**: bcrypt password hashing, comprehensive input validation
- **Error Handling**: Global exception handler with proper logging
- **API Structure**: Versioned endpoints under `/api/v1/`

### Frontend Structure (Currently Basic)
- Standard Vite + React setup
- TypeScript configuration with strict type checking
- ESLint configuration for code quality

### Environment Configuration
The backend uses environment variables loaded from `.env`:
- `SECRET_KEY`: JWT secret (minimum 32 characters)
- `MONGODB_URL`: MongoDB connection string
- `DATABASE_NAME`: MongoDB database name
- `DEBUG`: Enable debug mode
- `HOST`/`PORT`: Server configuration
- CORS settings for frontend communication

## Product Context

This application implements the AI PRD Generator described in `backend/docs/ai-prd-generator-prd.md`. Key features include:

1. **Chat-to-PRD Flow**: Users start with conversational ideation, then convert to structured PRDs
2. **Thinking Lens Framework**: Ensures coverage of Discovery, User Journey, Metrics, GTM, and Risks
3. **Dual Interaction Modes**: "Think" (brainstorming) and "Agent" (PRD updates)
4. **Workspace Interface**: Cursor-like UI with panels for project management
5. **AI Agent Orchestration**: LangGraph-based system for multiple specialized agents

## Development Guidelines

### Backend Development
- Follow FastAPI conventions with proper dependency injection
- Use async/await patterns consistently with Motor
- Implement proper error handling with specific HTTP status codes
- Write Pydantic schemas for all request/response models
- Add comprehensive tests for new endpoints
- Use the existing authentication middleware for protected routes

### Frontend Development  
- Use TypeScript strict mode
- Follow React hooks patterns
- Implement proper error boundaries
- Use environment variables for API endpoints
- Follow the existing ESLint configuration

### Database Considerations
- Use proper MongoDB indexing for performance
- Follow the existing user schema patterns in `models/user.py`
- Implement soft deletes where appropriate
- Use aggregation pipelines for complex queries

### Security Requirements
- Never log or commit sensitive information
- Use environment variables for all secrets
- Implement proper CORS configuration
- Follow password security requirements (8+ chars, mixed case, numbers, special chars)
- Validate all inputs through Pydantic schemas

## Testing Strategy
- Unit tests for services and utilities
- Integration tests for API endpoints
- Use pytest fixtures for database setup/teardown
- Mock external dependencies appropriately
- Test error conditions and edge cases