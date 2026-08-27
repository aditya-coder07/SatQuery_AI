FROM node:20-slim

WORKDIR /app

# Copy package files if they exist
COPY package.json* package-lock.json* ./
RUN if [ -f package.json ]; then npm install; fi

COPY . .

# Run next dev by default (assuming package.json has it, otherwise npx next dev)
CMD ["npm", "run", "dev"]
