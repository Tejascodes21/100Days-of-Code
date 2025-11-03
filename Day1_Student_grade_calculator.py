# --------------------------------------------
# 🧮 Student Grade Calculator
# Author: [Tejas Nikam]
# Day 1 - 100 Days of Code
# --------------------------------------------

# Function to calculate average marks
def calculate_average(marks):
    total = sum(marks)
    average = total/len(marks)
    return average

# Function to determine grade based on average
def calculate_grade(average):
    if average>=90:
        return "A+"
    elif average >=80:
        return "A"
    elif average>=70:
        return "B"
    elif average>=60:
        return "C"
    elif average>=50:
        return "D"
    else:
        return "F"

# Function to display results
def display_result(name, marks):
    average = calculate_average(marks)
    grade = calculate_grade(average)

    print("\n Student Result Summary")
    print("----------------------------------")
    print(f"Name: {name}")
    print(f"Marks: {marks}")
    print(f"Average: {average:.2f}")
    print(f"Grade: {grade}")
    print("----------------------------------")


def main():
    print("=== Student Grade Calculator ===")

    name = input("Enter student name: ")
    # input marks for 5 sub
    marks = []
    for i in range(5):
        while True:
            try:
                mark = float(input(f"Enter marks for subject {i+1} (out of 100): "))
                if 0 <= mark <= 100:
                    marks.append(mark)
                    break
                else:
                    print(" Please enter a valid mark between 0 and 100.")
            except ValueError:
                print("Invalid input! Please enter a numeric value.")

    # Display the results
    display_result(name, marks)


if __name__ == "__main__":
    main()
