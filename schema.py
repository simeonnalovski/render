from datetime import date, datetime
from typing import List

import pandas as pd
from pydantic import BaseModel, TypeAdapter, Field, constr, model_validator


class IncomeStatement(BaseModel):
    date: date
    symbol: str
    reportedCurrency: str
    cik: constr(pattern=r'^\d{10}$', strict=True)
    filingDate: date
    acceptedDate: datetime
    fiscalYear: int
    period: constr(pattern=r'^FY$',strict=True)
    revenue: int= Field(...,ge=0)
    costOfRevenue: int= Field(...,ge=0)
    grossProfit: int
    researchAndDevelopmentExpenses: int=Field(...,ge=0)
    generalAndAdministrativeExpenses: int=Field(...,ge=0)
    sellingAndMarketingExpenses: int=Field(...,ge=0)
    sellingGeneralAndAdministrativeExpenses: int=Field(...,ge=0)
    otherExpenses: int=Field(...,ge=0)
    operatingExpenses: int=Field(...,ge=0)
    costAndExpenses: int=Field(...,ge=0)
    netInterestIncome: int
    interestIncome: int=Field(...,ge=0)
    interestExpense: int=Field(...,ge=0)
    depreciationAndAmortization: int=Field(...,ge=0)
    ebitda: int
    ebit: int
    nonOperatingIncomeExcludingInterest: int
    operatingIncome: int
    totalOtherIncomeExpensesNet: int
    incomeBeforeTax: int
    incomeTaxExpense: int
    netIncomeFromContinuingOperations: int
    netIncomeFromDiscontinuedOperations: int
    otherAdjustmentsToNetIncome: int
    netIncome: int
    netIncomeDeductions: int
    bottomLineNetIncome: int
    eps: float
    epsDiluted: float
    weightedAverageShsOut: int
    weightedAverageShsOutDil: int

    @model_validator(mode="after")
    def validate_financial_calculations(self):
        if self.grossProfit != (self.revenue - self.costOfRevenue):
            raise ValueError("grossProfit isn't equal to revenue - costOfRevenue.")

        if self.operatingIncome != (self.grossProfit - self.operatingExpenses):
            raise ValueError("operatingIncome isn't equal to grossProfit - operatingExpenses.")

        if self.weightedAverageShsOutDil < self.weightedAverageShsOut:
            raise ValueError("weightedAverageShsOutDil cannot be less than weightedAverageShsOut.")

        if self.filingDate < self.acceptedDate.date():
            raise ValueError("filingDate must be on or after acceptedDate.")

        return self




# Validator for list of records
CSVValidator = TypeAdapter(List[IncomeStatement])


def validate_dataframe(df: pd.DataFrame):
        df = df.where(pd.notnull(df), None)  # replace NaN with None
        records = df.to_dict(orient="records")
        CSVValidator.validate_python(records)  # will raise if invalid


