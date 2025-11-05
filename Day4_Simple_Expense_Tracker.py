# Day 4 - Simple expense tracker
# Topics: Dictionaries, Functions, Loops, Conditionals

expenses = {}  # Dictionary to store category: total_amount

def add_expense():
    """Add a new expense under a specific category."""
    category = input("Enter category (e.g., Food, Travel, Bills): ").capitalize()
    try:
        amount = float(input("Enter amount:₹"))

        if category in expenses:
            expenses[category] += amount
        else:
            expenses[category] = amount
        print(f"Added ₹{amount} to {category} category.\n")
    except ValueError:
        print(" Please enter a valid number!\n")

def view_expenses():
    """Display all expenses and total spending."""
    if not expenses:
        print("No expenses recorded yet.\n")
        return

    print("\n------ Expense Summary ------")
    total = 0
    for category, amount in expenses.items():
        print(f"{category}: ₹{amount}")
        total += amount
    print("------------------------------")
    print(f"Total Expenses: ₹{total}\n")

def highest_expense():
    """Show the category with the highest expense."""
    if not expenses:
        print("No expenses yet!\n")
    else:
        highest = max(expenses, key=expenses.get)
        print(f" Highest Expense: {highest} - ₹{expenses[highest]}\n")

def reset_expenses():
    """Clear all expenses."""
    confirm = input("Are you sure you want to reset all expenses? (y/n): ").lower()
    if confirm == 'y':
        expenses.clear()
        print("All expenses cleared!\n")
    else:
        print("Reset cancelled.\n")

def main():
    while True:
        print("========== Expense Tracker ==========")
        print("1. Add Expense")
        print("2. View All Expenses")
        print("3. View Highest Expense")
        print("4. Reset All Expenses")
        print("5. Exit")

        choice = input("Choose an option (1-5): ")

        if choice == '1':
            add_expense()
        elif choice == '2':
            view_expenses()
        elif choice == '3':
            highest_expense()
        elif choice == '4':
            reset_expenses()
        elif choice == '5':
            print("Exiting Expense Tracker.")
            break
        else:
            print("Invalid choice! Please try again.\n")

if __name__ == "__main__":
    main()
