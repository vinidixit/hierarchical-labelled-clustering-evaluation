from .ticket_cleaning import clean_description, clean_subject
import pandas as pd

class TicketProcessor:

    def __init__(self, problem_tickets_df):
        self.problem_tickets_df = problem_tickets_df
        self.basic_sanitization()
        self.problem_df, self.subjects_df, self.descriptions_df = self.clean_data()

        self.subject_description_df = self.get_new_feature()


    def basic_sanitization(self):

        self.problem_tickets_df.subject_pr = self.problem_tickets_df.subject_pr.apply(
            lambda x: x.strip() if type(x) == str else '')

        self.problem_tickets_df.subject_ticket = self.problem_tickets_df.subject_ticket.apply(
            lambda x: x.strip() if type(x) == str else '')

        self.problem_tickets_df.description = self.problem_tickets_df.description.apply(
            lambda x: x.strip() if type(x) == str else '')

    def clean_data(self):
        problem_df = self.problem_tickets_df.subject_pr.drop_duplicates()
        subjects_df = self.problem_tickets_df.subject_ticket.drop_duplicates()
        descriptions_df = self.problem_tickets_df.description.drop_duplicates()

        subjects_df = subjects_df[subjects_df.map(len) > 0]
        descriptions_df = descriptions_df[descriptions_df.map(len) > 0]

        while True:
            initial_size = subjects_df.shape[0]
            subjects_df = subjects_df.apply(lambda x: clean_subject(x))
            subjects_df = subjects_df[subjects_df.map(len) > 0]
            final_size = subjects_df.shape[0]
            if initial_size == final_size:
                break

        while True:
            initial_size = descriptions_df.shape[0]
            descriptions_df = descriptions_df.apply(lambda x: clean_description(x))
            descriptions_df = descriptions_df[descriptions_df.map(len) > 0]
            final_size = descriptions_df.shape[0]

            if initial_size == final_size:
                break

        return problem_df, subjects_df, descriptions_df


    def get_new_feature(self):
        def join_texts(s1, s2):
            return s1 + '\n' + s2

        subject_description_df = self.subjects_df.combine(self.descriptions_df, join_texts, fill_value='')
        return subject_description_df

if __name__=='__main__':
    data_dir = '/Users/vdixit/Work/Data/processed_data/'
    data_file = data_dir + 'problem_tickets_consolidated.csv'

    problem_tickets_df = pd.read_csv(data_file)
    tp = TicketProcessor(problem_tickets_df)