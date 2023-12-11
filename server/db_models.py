from sqlalchemy import create_engine, Column, Integer, Text, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

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
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Transcript(source='{self.source}', episode='{self.episode}', segment={self.segment}, created_by={self.created_by}, created_at={self.created_at})>"

Base.metadata.create_all(engine)

