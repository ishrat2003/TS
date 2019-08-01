#######################################
# Dockerfile to launch Nginx
#######################################

# Set the base image
FROM ubuntu:18.04

# Package manager
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nginx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Update Nginx config
RUN echo "daemon off;" >> /etc/nginx/nginx.conf

# Update vhosts
ENV SITE jupyter
ADD ./docker/nginx/sites-available/${SITE} /etc/nginx/sites-available/${SITE}
RUN cd /etc/nginx/sites-enabled \
    && ln -s /etc/nginx/sites-available/${SITE} .

# Add www-data access rights
RUN chown -R www-data:www-data /var/www/

# Expose ports
EXPOSE 80

# Set the default command to execute
CMD ["nginx"]

# Docker does NOT support daemon services
# RUN systemctl enable nginx.service
