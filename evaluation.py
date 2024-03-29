from easydict import EasyDict
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


def evaluation(settings, test_loader, model):
    """performs model evaluation/testing
    Arguments:
        settings(EasyDict): easydict dictionary containing the evaluation settings
        test_loader(torch Dataloader): Dataloader used for Mini-batching
        model(torch.nn.Module): trained model to be evaluated
    returns:
    results(list): list of test accuracies
    """

    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_loader:
        x, y = x.to(settings.device), y.to(settings.device)

        _, y_pred = model(x).max(1)  # model prediction on clean examples

        report.nb_test += y.size(0)  # counts how many examples are in the batch
        report.correct += y_pred.eq(y).sum().item()  # counts how many examples in the batch are predicted correctly

        x_fgm = fast_gradient_method(model, x, 0.3, np.inf)
        _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples

        report.correct_fgm += y_pred_fgm.eq(y).sum().item()  # counts correctly predicted fgsm examples

        x_pgd = projected_gradient_descent(model, x, 0.3, 0.01, 40, np.inf)
        _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples

        report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # counts correctly predicted pgd examples

    results = EasyDict()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    results.clean = report.correct / report.nb_test

    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    results.fgsm_att = report.correct_fgm / report.nb_test
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )
    results.pgd_att = report.correct_pgd / report.nb_test

    return results
