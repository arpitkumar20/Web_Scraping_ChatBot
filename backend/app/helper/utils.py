import datetime
import bson
import decimal

class COMMON:
    def stringify(item):
        if isinstance(item, (bson.objectid.ObjectId,datetime.datetime,datetime.date,datetime.time,datetime.timezone,decimal.Decimal)):
            return str(item)
        
        return item