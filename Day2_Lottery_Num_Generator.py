import random
from datetime import datetime

def generate_numbers(total_numbers, min_value, max_value):
    """Generate unique random numbers for the lottery."""
    numbers = random.sample(range(min_value, max_value + 1), total_numbers)
    numbers.sort()
    return numbers

def save_numbers(numbers):
    """Save generated numbers to a file with a timestamp."""
    with open("lottery_results.txt", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} -> {numbers}\n")

def main():
    print(" Welcome to the Lottery Number Generator")

    while True:
        try:
            total_numbers = int(input("How many numbers do you want to generate? (e.g., 6): "))
            min_value = int(input("Enter the mini number in range: "))
            max_value = int(input("Enter the maxi number in range: "))

            if total_numbers > (max_value - min_value + 1):
                print("Oops! You’re asking for more unique numbers than the range allows.\n")
                continue

            lucky_numbers = generate_numbers(total_numbers, min_value, max_value)

            print("\nYour Lucky Numbers Are:")
            print(lucky_numbers)

            save_numbers(lucky_numbers)
            print("Saved in lottery_results.txt\n")

            again = input("Want to try again? (y/n): ").lower()
            if again != "y":
                print("\n Thanks for playing")
                break

        except ValueError:
            print("Please enter valid numbers\n")
       


if __name__ == "__main__":
    main()
