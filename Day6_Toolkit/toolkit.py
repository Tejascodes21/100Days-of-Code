# toolkit.py
# A simple personal utility toolkit using custom and built-in modules

import math_utils
import random
import os
from datetime import datetime

def show_menu():
    print("\n Personal Utility Toolkit")
    print("1. Factorial of a number")
    print("2. Prime check")
    print("3. Fibonacci sequence")
    print("4. Get current date & time")
    print("5. Generate random number")
    print("6. Show current directory")
    print("7. Exit")

while True:
    show_menu()
    try:
        choice = int(input("Enter your choice (1-7): "))
    except ValueError:
        print(" Please enter a valid number!")
        continue

    if choice == 1:
        n = int(input("Enter a number: "))
        print("Factorial:", math_utils.factorial(n))

    elif choice == 2:
        n = int(input("Enter a number: "))
        print(f"{n} is prime? →", math_utils.is_prime(n))

    elif choice == 3:
        n = int(input("Enter number of terms: "))
        print("Fibonacci sequence:", math_utils.fibonacci(n))

    elif choice == 4:
        print("Current Date & Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    elif choice == 5:
        start = int(input("Start range: "))
        end = int(input("End range: "))
        print("Random number:", random.randint(start, end))

    elif choice == 6:
        print("Current working directory:", os.getcwd())

    elif choice == 7:
        print("Exiting the toolkit.")
        break

    else:
        print("Invalid option! Try again.")
