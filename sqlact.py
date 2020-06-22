import pymysql


######################################

def add_to_sql(tablename, path, name):
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_helmet")

    cs3 = aiuse.cursor()
    sql3 = "insert into " + tablename + " values(%s, %s ,0 )"
    cs3.execute(sql3, (path, name))
    aiuse.commit()  # 提交数据
    cs3.close()  # 关闭游标cs3
    aiuse.close()  # 关闭数据库
    print("well done")


#################################
def delete_one_sql(tablename, image_id):
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_aiuse")

    cs3 = aiuse.cursor()
    sql = "delete from" + tablename + "where id=%s"
    cs3.execute(sql, (image_id))
    aiuse.commit()  ## 提交数据
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    print("well done")
######################################

def update_one_sql(tablename, name):
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_helmet")

    cs3 = aiuse.cursor()
    sql3 = "update " + tablename + " set  illegal=illegal+1 where path = %s"

    cs3.execute(sql3, (name))
    aiuse.commit()  ## 提交数据
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    print("well done")


###########################

def search_sql(tablename, image_id):
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_helmet")

    cs3 = aiuse.cursor()
    sql3 = "select * from " + tablename + " where image_id = %s "
    cs3.execute(sql3, (image_id))
    data3 = cs3.fetchall()  ## 指针移动到最后一行了
    # print(data3)
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    return data3


#############################


def search_by_path(tablename, path):
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_helmet")

    cs3 = aiuse.cursor()
    sql3 = "select * from " + tablename + " where path = %s "
    cs3.execute(sql3, (path))
    data3 = cs3.fetchall()  ## 指针移动到最后一行了
    # print(data3)
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    return data3


################################


def search_all_sql():
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_helmet")

    cs3 = aiuse.cursor()
    sql3 = "select name,illegal from face"
    cs3.execute(sql3)
    data3 = cs3.fetchall()  ## 指针移动到最后一行了
    # print(data3)
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    return data3

def delete_illegal():
    aiuse = pymysql.connect(host="localhost",
                            port=3306,
                            user="root",
                            password="wang518518",
                            db="db_aiuse")

    cs3 = aiuse.cursor()
    sql = "delete from" + "where id=%s"
    cs3.execute(sql, ())
    aiuse.commit()  ## 提交数据
    cs3.close()  ## 关闭游标cs3
    aiuse.close()  ## 关闭数据库
    print("well done")