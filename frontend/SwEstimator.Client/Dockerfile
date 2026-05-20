# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY SwEstimator.Client.csproj ./
RUN dotnet restore

COPY . ./
RUN dotnet publish -c Release -o /app/publish

# ── Stage 2: serve with nginx ────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Replace the default nginx config with one suited for Blazor WASM SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy published Blazor artifacts
COPY --from=build /app/publish/wwwroot /usr/share/nginx/html

EXPOSE 80
