# syntax=docker/dockerfile:1.7
# Optional CA bundle for TLS-intercepting proxies (see backend/Dockerfile).
FROM scratch AS certs

FROM node:20-alpine AS build
COPY --from=certs / /certs/
WORKDIR /app
COPY package.json package-lock.json ./
RUN CA=/certs/ca-certificates.crt; if [ -f "$CA" ]; then export NODE_EXTRA_CA_CERTS=$CA; fi; \
    npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
