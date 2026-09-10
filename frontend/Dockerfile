FROM node:22-alpine AS build
WORKDIR /src
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.29-alpine
# 官方 entrypoint 仅展开这两个部署变量，避免替换 Nginx 的 $host 等运行时变量。
ENV NGINX_ENVSUBST_FILTER='^(API_UPSTREAM|NGINX_CLIENT_MAX_BODY_SIZE)$'
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
