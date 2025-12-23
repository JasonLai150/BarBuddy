# BarBuddy Frontend

A React Native/Expo frontend for the BarBuddy application.

## Getting Started

### Prerequisites
- Node.js 18+ installed
- Expo CLI: `npm install -g expo-cli`

### Installation

```bash
cd frontend
npm install
```

### Running the App

**Web (Browser):**
```bash
npm run web
```

**iOS (requires macOS):**
```bash
npm run ios
```

**Android (requires Android SDK):**
```bash
npm run android
```

**Start in Development Mode:**
```bash
npm start
```

## Project Structure

```
frontend/
├── App.tsx           # Main app component
├── app.json          # Expo configuration
├── package.json      # Dependencies
├── tsconfig.json     # TypeScript configuration
└── src/              # Source files (to be created)
    ├── screens/      # Screen components
    ├── components/   # Reusable components
    ├── services/     # API services
    └── types/        # TypeScript types
```

## Configuration

Update the API endpoint in your services to point to your AWS CDK backend:

```typescript
// src/services/api.ts
const API_BASE_URL = 'https://your-api-endpoint.com';
```

## Learn More

- [Expo Documentation](https://docs.expo.dev)
- [React Native Documentation](https://reactnative.dev)
- [TypeScript Documentation](https://www.typescriptlang.org)
