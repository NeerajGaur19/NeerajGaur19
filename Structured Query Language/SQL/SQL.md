
SQL (Structured Query Language) is the standard language used to store, retrieve, manipulate, and manage data in relational databases.

Think of SQL as the language you use to communicate with a database.


# Main Categories of SQL Commands

## 1. DDL (Data Definition Language)

Used to create and modify database objects.

    Command	                              Purpose
    CREATE	                        Create a table or database
    ALTER	                        Modify a table
    DROP	                        Delete a table
    TRUNCATE	                    Remove all rows (keeps table structure)
    RENAME	                        Rename an object

### Example:

    CREATE TABLE Employees
    (
        EmpID INT,
        Name VARCHAR(50),
        Salary DECIMAL(10,2)
    );
    

## 2. DML (Data Manipulation Language)

Used to insert, update, and delete data.
    
    Command	            Purpose
    INSERT	            Add data
    UPDATE	            Modify data
    DELETE	            Remove data

### Example:
    
    INSERT INTO Employees
        VALUES (101, 'Neha', 70000);
    
    UPDATE Employees
        SET Salary = 75000
        WHERE EmpID = 101;
    
    DELETE FROM Employees
        WHERE EmpID = 101;

    
## 3. DQL (Data Query Language)

Used to retrieve data.

The primary command is:

    SELECT * FROM Employees;

### Examples:

    SELECT Name, Salary
    FROM Employees;

    SELECT *
    FROM Employees
    WHERE Salary > 60000;

## 4. DCL (Data Control Language)

Used to control access.

    GRANT
    REVOKE

## 5. TCL (Transaction Control Language)

Used to manage transactions.

* COMMIT
* ROLLBACK
* SAVEPOINT

### Example Queries

Retrieve specific columns:

    SELECT Name, Age
    FROM Students;
