import sqlite3
import json

connection = sqlite3.connect('../Database/Phase_3.db')
c = connection.cursor()

Alter_Query1 = "ALTER TABLE data ADD All_Label VARCHAR(20);"
c.execute(Alter_Query1)
Alter_Query2 = "ALTER TABLE data ADD Is_Target BOOLEAN;"
c.execute(Alter_Query2)
print("Altered Table")

Update_Target_Query = "UPDATE data SET Is_Target = TRUE WHERE videoID <= 2872;"
c.execute(Update_Target_Query)
print("Updated Target")
Update_Non_Target_Query = "UPDATE data SET Is_Target = FALSE WHERE videoID > 2872;"
c.execute(Update_Non_Target_Query)
print("Updated Non-Target")

with open('../Database/category_map_all.json', 'r') as f:
    label_data = json.load(f)

i = 0
for x in label_data:
    # print(x, label_data[x])
    Update_Label_Query = f"UPDATE data SET All_Label = '{label_data[x]}' WHERE Video_Name = '{x}';"
    c.execute(Update_Label_Query)
    i += 1
    if i % 100 == 0:
        print(i, " Videos labels updated")
print("\nUpdated All Labels")

# Verify if All_Label and Action_Label is not matching
Verify_Query = "SELECT count(*) from data where All_Label != Action_Label AND Action_Label != NULL;"
c.execute(Verify_Query)

rows = c.fetchall()
print(rows[0])

commit = input("Commit Changes to db?(y/N) ")

if commit == 'y':
    connection.commit()

connection.close()
