from django.urls import path

from apps.fl import views

urlpatterns = [
    path('startTrain_singleuser/<str:dataset_title>', views.startTrain_singleuser),
    path('startTrain_multiuser/<str:task_id>', views.startTrain_multiuser),
    path('startTrain_fedmdh/<str:task_id>', views.startTrain_fedmdh),
    path('get_single_result/<str:task_id>/<str:username>', views.get_single_result),
    path('get_multi_result/<str:task_id>', views.get_multi_result),
    path('get_single_result_fedmdh/<str:task_id>/<str:username>', views.get_single_result_fedmdh),
    path('get_multi_result_fedmdh/<str:task_id>', views.get_multi_result_fedmdh),
]
