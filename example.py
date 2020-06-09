import sys
import mlflow

if __name__ == '__main__':
    message = sys.argv[1] if len(sys.argv) > 1 else "No message supplied"
    with mlflow.start_run():
        print(message)
