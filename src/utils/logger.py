def log_message(log_file, message):
    """Log message to file and print to console"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
