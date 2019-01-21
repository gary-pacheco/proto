FROM centos:7

# Install OS packages
RUN yum -y update

# Create a director for the application
RUN mkdir /app
WORKDIR /app

# Copy Python package requirements
COPY ./requirements.txt /app/requirements.txt

# Install Python & packages
# + then install app packages from requirements
# - then remove unnecessary packages
RUN yum install -y epel-release
RUN yum install -y python36 python36-devel python36-setuptools && \
 ln -s /usr/bin/python3.6 /usr/bin/python3 && \
 python3 -m ensurepip && \
 python3 -m pip install --upgrade pip && \
 pip3 --no-cache-dir install -r requirements.txt && \
 yum remove -y epel-release iputils perl && \
 yum autoremove -y && \
 yum clean all && \
 rm -rd /tmp/*

# Copy source code
COPY ./ /app

# Run application
CMD ["python3", "/app/app.py"]

EXPOSE 80/tcp
