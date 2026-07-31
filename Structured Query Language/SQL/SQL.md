
SQL (Structured Query Language) is the standard language used to store, retrieve, manipulate, and manage data in relational databases.

Think of SQL as the language you use to communicate with a database.


# Main Categories of SQL Commands

## 1. DDL (Data Definition Language)

Used to create and modify database objects.

    Command	                              Purpose
    CREATE	                        Create a table or database
    ALTER	                          Modify a table
    DROP	                          Delete a table
    TRUNCATE	                      Remove all rows (keeps table structure)
    RENAME	                        Rename an object

### Example:

    CREATE TABLE Employees
    (
        EmpID INT,
        Name VARCHAR(50),
        Salary DECIMAL(10,2)
    );
    
