import os
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def all_gs_cvc(dict_plot_gs, cv_results, best_params, best_score, dir_html=None):

    list_sample = dict_plot_gs['list_sample']
    list_discrete_parmas = dict_plot_gs['list_discrete_parmas']
    list_const_parmas = dict_plot_gs['list_const_parmas']
    dict_scorer = dict_plot_gs['dict_scorer']

    df_perf = pd.DataFrame(cv_results['params'])

    # assign train test score
    for sample in list_sample:
        for k_scoring, scoring in dict_scorer.items():
            key_mean = f'mean_{sample}_{k_scoring}'
            key_std = f'std_{sample}_{k_scoring}'

            df_perf[key_mean] = cv_results[key_mean]
            df_perf[key_std] = cv_results[key_std]

    # assign color of each discrete combination
    dict_color = define_parmas_color(dict_scorer, df_perf, list_discrete_parmas, list_sample)

    for const_parmas in list_const_parmas:

        best_fix_const_parmas = best_params[const_parmas]

        # filter out other combination and remain a specific const parmas performance
        dict_filter = {k: v for k, v in best_params.items() if k not in [const_parmas] + list_discrete_parmas}
        df_const_parmas = df_perf.copy()
        for k, v in dict_filter.items():
            df_const_parmas = df_const_parmas[df_const_parmas[k] == v]

        print(f'const_parmas: {const_parmas}')
        fig = gs_cvc(df_const_parmas, dict_color, list_sample, dict_scorer, list_discrete_parmas, const_parmas,
                     best_score, best_fix_const_parmas)

        if dir_html is not None:
            Path(dir_html).mkdir(parents=True, exist_ok=True)
            df_const_parmas.to_csv(os.path.join(dir_html, f'const_parmas_{const_parmas}.csv'), index=False)
            fig.write_html(os.path.join(dir_html, f'const_parmas_{const_parmas}.html'))
        else:
            fig.show()

        print()


def define_parmas_color(dict_scoring, df_perf, list_discrete_parmas, list_sample,
                        list_color=['rgb(255,0,0,0)', 'rgb(255,128,0,0)', 'rgb(204,204,0,0)', 'rgb(0,255,0,0)',
                                    'rgb(0,255,255,0)', 'rgb(0,0,255,0)', 'rgb(127,0,255,0)', 'rgb(255,0,127,0)',
                                    'rgb(255,153,153,0)', 'rgb(255,255,153,0)', 'rgb(153,255,153,0)',
                                    'rgb(224,224,224,0)'

                                    ]):
    """assign different color in different parameter and metrics"""

    # define color of each parmas
    dict_color = {}
    index_color = 0

    for k_scoring, scoring in dict_scoring.items():
        for discrete_parmas, _ in df_perf.groupby(list_discrete_parmas):
            for sample in list_sample:
                str_parmas = f'{k_scoring}_{sample}_{discrete_parmas}'
                dict_color[str_parmas] = list_color[index_color]

            index_color += 1

    return dict_color


def gs_cvc(df_spec_const_parama, dict_color, list_sample, dict_scoring, list_discrete_parmas, col_x,
           max_test_score, max_test_score_x):
    """plot one grid search cross validation curve in a df which contain specific const parmas result"""

    fig = go.Figure()

    for sample in list_sample:
        for k_scoring, scoring in dict_scoring.items():

            key_mean = f'mean_{sample}_{k_scoring}'
            key_std = f'std_{sample}_{k_scoring}'

            for discrete_parmas, df_spec_perf in df_spec_const_parama.groupby(list_discrete_parmas):
                str_parmas = f'{k_scoring}_{sample}_{discrete_parmas}'

                # plot value
                fig.add_trace(go.Scatter(
                    x=df_spec_perf[col_x], y=df_spec_perf[key_mean],
                    line_color=dict_color[str_parmas],
                    name=str_parmas,
                    line=dict(dash=None if sample == 'train' else 'dot')
                ))

                # plot upper and lower
                if sample == 'test':
                    series_upper = df_spec_perf[key_mean] + df_spec_perf[key_std]
                    series_lower = df_spec_perf[key_mean] - df_spec_perf[key_std]

                    y = pd.concat([series_upper, series_lower.iloc[::-1]])
                    x = pd.concat([df_spec_perf[col_x], df_spec_perf[col_x].iloc[::-1]])

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        fill='toself',
                        line_color=dict_color[str_parmas].replace('rgb', 'rgba'),
                        fillcolor=dict_color[str_parmas].replace(',0)', ',0.05)').replace('rgb', 'rgba'),
                        name='filled' + str_parmas,
                        showlegend=False,
                    ))

    # plot best score and it parameter
    fig.add_vline(x=max_test_score_x, line_width=3, line_dash="dash", line_color="black")

    fig.add_trace(go.Scatter(
        x=[max_test_score_x], y=[max_test_score],
        mode='markers', name='Best selected score',
        marker_symbol='x-open',
        marker=dict(
            color='black',
            size=30,
        )
    )
    )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=800,
        hoverlabel=dict(namelength=40))

    fig.update_xaxes(title_text=col_x)
    fig.update_yaxes(title_text='score')

    return fig