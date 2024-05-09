import mysql.connector

# ### SQL Connection ###

def connect_to_database():
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host="87.140.18.238",
            port=30,
            user="admin",
            password="dimensify",
            database="dimensify",
            autocommit=True 
        )

        if connection.is_connected():
            print("Connected to MySQL database")
            return connection

    except mysql.connector.Error as error:
        print("Error connecting to MySQL database:", error)
        return None

def insert_data_to_user_library(connection, gif_location, model_location, task_id, zip_location, user_id):
    try:
        if connection:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            data = {
                "gif_location": gif_location,
                "model_location": model_location,
                "task_id": task_id,
                "zip_location": zip_location,
                "user_id": user_id  # Assuming user_id is an integer
            }

            # SQL query to insert data into the user_library table
            insert_query = """
            INSERT INTO user_library (gif_location, model_location, task_id, zip_location, user_id)
            VALUES (%s, %s, %s, %s, %s)
            """

            # Execute the SQL query with the data
            cursor.execute(insert_query, (
                data["gif_location"],
                data["model_location"],
                data["task_id"],
                data["zip_location"],
                data["user_id"]
            ))

            # Commit the transaction
            connection.commit()
            
            print("Data inserted successfully!")
            
            # return data["id"]
            # Get the last inserted primary key ID
            last_inserted_id = cursor.lastrowid

            # Return the last inserted primary key ID
            return last_inserted_id


    except mysql.connector.Error as error:
        print("Error inserting data into MySQL table:", error)

    finally:
        # Close the cursor
        if 'cursor' in locals():
            cursor.close()


def insert_data_to_model_details(connection, fidelity, file_format, time_taken, input_text, user_library_id):
    try:
        if connection:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            data = {
                "fidelity": fidelity,
                "file_format": file_format,
                "gen_text": input_text,
                "time_taken": time_taken,
                "common_library_id": None,
                "user_library_id": user_library_id
            }

            # SQL query to insert data into the model_details table
            insert_query = """
            INSERT INTO model_details (fidelity, file_format, gen_text, time_taken, common_library_id, user_library_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """

            # Execute the SQL query with the data
            cursor.execute(insert_query, (
                data["fidelity"],
                data["file_format"],
                data["gen_text"],
                data["time_taken"],
                data["common_library_id"],
                data["user_library_id"]
            ))
            # Commit the transaction
            connection.commit()

            # # Return the last inserted primary key
            # return data["id"]
        
            # Get the last inserted primary key ID
            last_inserted_id = cursor.lastrowid

            # Return the last inserted primary key ID
            return last_inserted_id

    except mysql.connector.Error as error:
        print("Error inserting data into MySQL table:", error)

    finally:
        # Close the cursor
        if 'cursor' in locals():
            cursor.close()

def get_user_id_by_email(connection, email):
    '''
    Retrieve the user ID from the users table based on the email.

    Parameters
    ----------
    connection : mysql.connector.connection.MySQLConnection
        MySQL database connection.
    email : str
        Email of the user.

    Returns
    -------
    int or None
        User ID if found, None otherwise.
    '''
    try:
        if connection:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            # SQL query to select the id from the users table where the email matches
            select_query = """
            SELECT id FROM users WHERE email = %s
            """

            # Execute the SQL query with the email parameter
            cursor.execute(select_query, (email,))

            # Fetch the result
            result = cursor.fetchone()

            if result:
                # If a user with the given email is found, return the user ID
                return result[0]
            else:
                # If no user is found with the given email, return None
                return None

    except mysql.connector.Error as error:
        print("Error retrieving user ID from MySQL table:", error)

    finally:
        # Close the cursor
        if 'cursor' in locals():
            cursor.close()