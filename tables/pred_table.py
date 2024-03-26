from config_db.database import Base
from sqlalchemy import Column, Integer, Float, String

class PredTable(Base):
    __tablename__ = 'predictions_table'
    id = Column(Integer, primary_key=True)
    month_salary = Column(Float)
    bank_accounts = Column(Float)
    num_credit_card = Column(Float)
    interest_rate = Column(Float)
    delay_date = Column(Float)
    credit_inquiries = Column(Float)
    credit_utilization = Column(Float)
    emi_per_month = Column(Float)
    age = Column(Float)
    annual_income = Column(Float)
    num_loan = Column(Float)
    delay_payment = Column(Float)
    chaged_limit = Column(Float)
    outstanding_debt = Column(Float)
    amount_invested = Column(Float)
    month_balance = Column(Float)
    occupation = Column(String)
    history_age = Column(Float)
    paymen_min = Column(String)
    payment_behaviour = Column(String)
    payday_loan = Column(Float)
    personal_loan = Column(Float)
    mortgage_loan = Column(Float)
    student_loan = Column(Float)
    auto_loan = Column(Float)
    credit_builder_loan = Column(Float)
    home_equity_loan = Column(Float)
    debt_cons_loan = Column(Float)

    prediction = Column(String)