from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import datetime
import json
from json import JSONEncoder
import os

engine = create_engine(os.environ['DB_CONNECTION_STRING'])  # Replace with your database URI
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Transcript(Base):
    __tablename__ = 'transcripts'

    id = Column(Integer, primary_key=True)
    source = Column(Text, nullable=False, index=True)
    episode = Column(Text, nullable=False, index=True)
    segment = Column(Integer, nullable=False, index=True)
    created_by = Column(Text, nullable=False, index=True)
    data = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Transcript(source='{self.source}', episode='{self.episode}', segment={self.segment}, created_by={self.created_by}, created_at={self.created_at})>"
    

Base.metadata.create_all(engine)

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()

        return JSONEncoder.default(self, obj)

def to_dict(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def table_to_json_str(table):
    entries = []

    for e in Session().query(table).yield_per(100):
        entries.append(to_dict(e))

    return json.dumps(entries, indent=4, cls=CustomJSONEncoder)
