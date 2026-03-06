FROM node:20-alpine
RUN apk add --no-cache python3 g++
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 3050
CMD ["node", "server.js"]
