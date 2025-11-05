from datetime import datetime

def write_note():
    """Write new note and save it to file."""
    note = input("\nEnter your note: ")
    if not note.strip():
        print("Empty note not saved!")
        return
    
    try:
        with open("notes.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp} - {note}\n")
        print("Note saved successfully!\n")
    except Exception as e:
        print(f"Error saving note: {e}\n")


def view_notes():
    """Display all saved notes."""
    try:
        with open("notes.txt", "r") as file:
            notes = file.readlines()
            if not notes:
                print("\nNo notes found!\n")
                return
            print("\n Your Notes:\n")
            for note in notes:
                print(note.strip())
            print()
    except FileNotFoundError:
        print("\nNo notes file found. Add your first note!\n")
    except Exception as e:
        print(f"Error reading notes: {e}\n")


def delete_notes():
    """Delete all notes safely."""
    confirm = input("\nAre you sure you want to delete all notes? (y/n): ").lower()
    if confirm != 'y':
        print("Deletion cancelled.\n")
        return

    try:
        open("notes.txt", "w").close()  # Clears the file
        print("All notes deleted successfully!\n")
    except Exception as e:
        print(f"Error deleting notes: {e}\n")


def main():
    print("Welcome to the Personal Notes Keeper")
    
    while True:
        print("\nChoose an option:")
        print("1 Write a new note")
        print("2 View all notes")
        print("3 Delete all notes")
        print("4 Exit")

        try:
            choice = int(input("\nEnter your choice (1-4): "))
            
            if choice == 1:
                write_note()
            elif choice == 2:
                view_notes()
            elif choice == 3:
                delete_notes()
            elif choice == 4:
                print("\nThanks for using Notes Keeper! Goodbye!\n")
                break
            else:
                print("Please enter a number between 1 and 4.\n")
        
        except ValueError:
            print("Invalid input! Please enter a number.\n")
        except Exception as e:
            print(f"Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
