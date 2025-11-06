# Bank Account Simulator
# Concepts covered: Classes, Inheritance, Encapsulation, Methods, Input Validation

class BankAccount:
    def __init__(self, account_number, holder_name, balance=0):
        self.account_number = account_number
        self.holder_name = holder_name
        self._balance = balance  # Protected attribute

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            print(f"Deposited ₹{amount}. New balance: ₹{self._balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount > self._balance:
            print("Insufficient funds.")
        elif amount <= 0:
            print("Withdrawal amount must be positive.")
        else:
            self._balance -= amount
            print(f"Withdrawn ₹{amount}. Remaining balance: ₹{self._balance}")

    def check_balance(self):
        print(f"Current balance: ₹{self._balance}")

    def __str__(self):
        return f"Account[{self.account_number}] - Holder: {self.holder_name}, Balance: ₹{self._balance}"


# Child Class demonstrating Inheritance
class SavingsAccount(BankAccount):
    def __init__(self, account_number, holder_name, balance=0, interest_rate=0.05):
        super().__init__(account_number, holder_name, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        print(f"Interest of ₹{round(interest, 2)} added. New balance: ₹{round(self._balance, 2)}")

def main():
    print("Welcome to Bank simulator!")
    acc_num = input("Enter account number: ")
    name = input("Enter account holder name: ")

    account = SavingsAccount(acc_num, name, balance=1000)

    while True:
        print("\nChoose an operation:")
        print("1 Deposit Money")
        print("2 Withdraw Money")
        print("3 Check Balance")
        print("4 Add Interest")
        print("5 Account Details")
        print("6 Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            amount = float(input("Enter deposit amount: "))
            account.deposit(amount)
        elif choice == '2':
            amount = float(input("Enter withdrawal amount: "))
            account.withdraw(amount)
        elif choice == '3':
            account.check_balance()
        elif choice == '4':
            account.add_interest()
        elif choice == '5':
            print(account)
        elif choice == '6':
            print("Thank you for banking with us!")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
