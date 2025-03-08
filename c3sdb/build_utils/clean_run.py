# python3 -m c3sdb.build_utils.clean_run

import sqlite3
from c3sdb.build_utils.mqns import add_mqns_to_db

def _main():
    clean_con = sqlite3.connect("C3S_clean.db")
    clean_cur = clean_con.cursor()
    # add MQNs to clean databse
    print("adding MQNs to cleaned database entries ...")
    n_mqns_clean = add_mqns_to_db(clean_cur)
    print(f"\tentries with MQNs in clean database: {n_mqns_clean}")
    # commit changes to clean database and close
    clean_con.commit()
    clean_con.close()

if __name__ == "__main__":
    _main()