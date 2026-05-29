FROM rasa/rasa:3.6.0-full

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command: run Rasa
CMD ["run", "--enable-api", "--cors", "*"]
