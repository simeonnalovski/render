from datetime import date, datetime
from typing import List

import pandas as pd
from pydantic import BaseModel, TypeAdapter, Field, constr, model_validator, conint


class IncomeStatement(BaseModel):
    date: date
    symbol: str
    reportedCurrency: str
    cik: constr(pattern=r'^\d{10}$', strict=True)
    filingDate: date
    acceptedDate: datetime
    fiscalYear: int
    period: constr(pattern=r'^FY$',strict=True)
    revenue: conint(ge=0)
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

        if self.date > self.filingDate:
            raise ValueError("statement date cannot be after filing date.")

        if self.fiscalYear != self.date.year:
            raise ValueError("fiscalYear must match the year of the report date.")

        if self.ebitda != (self.ebit + self.depreciationAndAmortization):
            raise ValueError("EBITDA must equal EBIT + Depreciation & Amortization.")

        if self.costAndExpenses != (self.costOfRevenue + self.operatingExpenses):
            raise ValueError("costAndExpenses must equal costOfRevenue + operatingExpenses.")

        if self.netInterestIncome != (self.interestIncome - self.interestExpense):
            raise ValueError("netInterestIncome must equal interestIncome - interestExpense.")

        if self.grossProfit < 0:
            raise ValueError("grossProfit cannot be negative.")

        if self.netIncome > self.revenue:
            raise ValueError("netIncome cannot exceed revenue.")

        return self




# Validator for list of records
CSVValidator = TypeAdapter(List[IncomeStatement])


def validate_dataframe(df: pd.DataFrame):
        df = df.where(pd.notnull(df), None)  # replace NaN with None
        records = df.to_dict(orient="records")
        CSVValidator.validate_python(records)  # will raise if invalid


