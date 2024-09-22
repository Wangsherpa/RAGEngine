import sys

sys.path.append(".")
from src.generation import generate_response


def main():
    context = "Wang is a data scientist who works at Amphora Software along with Piyush and Abhishek. Saipriya, Rishitha and Mahuri also work in the same company."
    query = "Who works at amphora?"
    answer = generate_response(query=query, context=context)
    print(answer)


if __name__ == "__main__":
    main()
