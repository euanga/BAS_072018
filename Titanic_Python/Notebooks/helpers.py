import os
from azureml.core.workspace import Workspace
from azureml.core.project import Project
from azureml.core.run import Run

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def get_run_history_url_2(run):
    url = 'https://mlworkspace.azureml-test.net/home/%2Fsubscriptions%2F{0}%2FresourceGroups%2F{1}%2Fproviders%2FMicrosoft.MachineLearningServices%2Fworkspaces%2F{2}/projects/{3}/run-history/run-details/{4}'
    rh = run.history
    wso = rh.workspace_object
    return url.format(wso.subscription_id, 
                      wso.resource_group, 
                      wso.name,
                      rh.name,
                      run.id)


def get_run_history_url(run):
    url = 'https://mlworkspace.azureml-test.net/portal/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.MachineLearningServices/workspaces/{2}/history/{3}/runs/{4}'    
    rh = run.history
    wso = rh.workspace_object
    return url.format(wso.subscription_id, 
                      wso.resource_group, 
                      wso.name,
                      rh.name,
                      run.id)
