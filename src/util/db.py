import psycopg2
#from vo import experiment_vo

conn = psycopg2.connect(host='localhost', dbname='postgres', user='postgres', password='1', port='5432')
cur = conn.cursor()

def ins_experiment(experiment_vo):
    print(">>>> ins_experiment ",experiment_vo.score)
    # cur.execute("""
    # INSERT    INTO experiment (work_dtm9, scenario_id) VALUES ('2022-02-20', 'a2')
    # """)
    # conn.commit()
 
# ins_experiment()

